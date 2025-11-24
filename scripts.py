# scripts.py
# Uploads processed ML-refined metrics to Firestore

import os
import pandas as pd
from typing import Dict, Set
from google.cloud import firestore
from google.oauth2 import service_account

PROJECT_ID = os.getenv("GCP_PROJECT", "pactrack-d63e9")
KEY_PATH   = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/Users/neilkhandelwal/Downloads/svc-key.json")
OUTDIR     = os.getenv("OUTDIR", "data/processed")
ML_VERSION = os.getenv("ML_VERSION", "v1")

RACES_CSV = f"{OUTDIR}/by_race.csv"
CANDS_CSV = f"{OUTDIR}/by_candidate.csv"
REFINE_SUM = f"{OUTDIR}/core_channel_refine_summary.csv"

def fold_channel_mix(row: pd.Series) -> Dict[str, float]:
    blocklist = {
        "pct_other_unclear",
        "pct_media_placement",
        "pct_consulting_strategy",
        "pct_fundraising_compliance",
        "pct_legal_accounting",
        "pct_events_travel",
        # variants that sometimes slip in:
        "pct_other",
        "pct_unknown",
        "pct_other_unclear ",
    }
    mix = {}
    for k, v in row.items():
        if not isinstance(k, str): 
            continue
        if not k.startswith("pct_"):
            continue
        if k in blocklist:
            continue
        if pd.notna(v):
            try:
                mix[k] = float(v)
            except Exception:
                pass
    return mix

def drop_pct_columns(d: dict) -> dict:
    for k in list(d.keys()):
        if isinstance(k, str) and k.startswith("pct_"):
            d.pop(k, None)
    return d

def to_doc(row: pd.Series, key_col: str, source_csv: str) -> dict:
    # Critical metadata fields that must be preserved even if None
    # These are essential for the website to display properly
    METADATA_FIELDS = {
        "office", "state", "district", "state_code",  # Candidate & race metadata
        "candidates",  # Race -> candidates list
        "candidate_name", "candidate_id",  # Candidate identifiers
        "race"  # Race identifier
    }
    
    # Convert to dict without dropping NaN first
    d = row.to_dict()
    
    # Replace NaN/None with appropriate values for non-metadata fields
    # Keep metadata fields even if they're None (for JSON compatibility)
    cleaned = {}
    for k, v in d.items():
        # Handle arrays/lists separately (pd.notna doesn't work on them)
        if isinstance(v, (list, tuple)):
            cleaned[k] = v
        elif pd.api.types.is_scalar(v) and pd.notna(v):
            # Keep non-NaN scalar values as-is
            cleaned[k] = v
        elif k in METADATA_FIELDS:
            # Preserve metadata fields even if None (stored as null in Firestore)
            cleaned[k] = None
        # Otherwise drop the field (old behavior for non-critical fields)
    
    # Extract channel mix
    mix = fold_channel_mix(row)
    
    # Remove pct_ columns (they're consolidated into channel_mix)
    cleaned = drop_pct_columns(cleaned)
    
    # Add channel mix if present
    if mix:
        cleaned["channel_mix"] = mix
    
    # Add metadata
    cleaned["ml_version"] = ML_VERSION
    cleaned["source_csv"] = os.path.basename(source_csv)
    
    return cleaned

def validate_metadata(df: pd.DataFrame, key_col: str, col_name: str) -> None:
    """Validate that critical metadata fields are present in the DataFrame."""
    if col_name == "races":
        required = ["office", "state", "district"]
        missing = [f for f in required if f not in df.columns]
        if missing:
            print(f"⚠️ WARNING: {col_name} missing metadata fields: {missing}")
        else:
            # Count how many rows have metadata
            has_metadata = df[required].notna().all(axis=1).sum()
            total = len(df)
            print(f"ℹ️ {col_name}: {has_metadata}/{total} rows have complete metadata")
    elif col_name == "candidates":
        required = ["office", "district"]
        state_col = "state_code" if "state_code" in df.columns else "state"
        if state_col in df.columns:
            required.append(state_col)
        missing = [f for f in required if f not in df.columns]
        if missing:
            print(f"⚠️ WARNING: {col_name} missing metadata fields: {missing}")
        else:
            has_metadata = df[required].notna().all(axis=1).sum()
            total = len(df)
            print(f"ℹ️ {col_name}: {has_metadata}/{total} rows have complete metadata")

def upsert_collection(df: pd.DataFrame, key_col: str, col_name: str, db: firestore.Client, source_csv: str, purge: bool = False):
    # Validate metadata before upload
    validate_metadata(df, key_col, col_name)
    
    existing_ids: Set[str] = set()
    if purge:
        for d in db.collection(col_name).stream():
            existing_ids.add(d.id)
    new_ids: Set[str] = set(str(x) for x in df[key_col].astype(str).tolist())

    batch = db.batch()
    count, BATCH = 0, 450
    metadata_sample_logged = False
    
    for _, row in df.iterrows():
        doc_id = str(row[key_col])
        data = to_doc(row, key_col, source_csv)
        ref = db.collection(col_name).document(doc_id)
        batch.set(ref, data, merge=True)
        count += 1
        
        # Log first document's metadata for debugging
        if not metadata_sample_logged and count == 1:
            metadata_fields = {k: v for k, v in data.items() if k in ["office", "state", "district", "state_code"]}
            if metadata_fields:
                print(f"ℹ️ Sample {col_name} metadata for {doc_id}: {metadata_fields}")
            metadata_sample_logged = True
        
        if count % BATCH == 0:
            batch.commit(); batch = db.batch()
    batch.commit()
    print(f"✅ Upserted {count} docs into {col_name}")

    if purge:
        stale = existing_ids - new_ids
        if stale:
            print(f"⚠️ Purging {len(stale)} stale docs in {col_name}")
            batch = db.batch()
            for i, doc_id in enumerate(stale, 1):
                batch.delete(db.collection(col_name).document(doc_id))
                if i % BATCH == 0:
                    batch.commit(); batch = db.batch()
            batch.commit()

def add_meta(db: firestore.Client):
    summary = {}
    if os.path.exists(REFINE_SUM):
        try:
            s = pd.read_csv(REFINE_SUM)
            summary = {r["metric"]: r["value"] for _, r in s.iterrows()}
        except Exception:
            pass
    db.collection("meta").document("app").set({
        "last_updated": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ml_version": ML_VERSION,
        "source": {
            "races_csv": RACES_CSV,
            "cands_csv": CANDS_CSV
        },
        "refinement_summary": summary
    }, merge=True)

def main():
    creds = service_account.Credentials.from_service_account_file(KEY_PATH)
    db = firestore.Client(project=PROJECT_ID, credentials=creds)

    races = pd.read_csv(RACES_CSV)
    cands = pd.read_csv(CANDS_CSV)

    assert "race" in races.columns, "races CSV missing 'race' column"
    assert "candidate_id" in cands.columns, "candidates CSV missing 'candidate_id' column"

    # Note: candidates list is now already in by_race.csv from process_data.py
    if "candidates" in races.columns:
        import ast
        races["candidates"] = races["candidates"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)

    upsert_collection(races, "race", "races", db, RACES_CSV, purge=False)
    upsert_collection(cands, "candidate_id", "candidates", db, CANDS_CSV, purge=False)

    # Upload operating expenditures
    pac_csv = f"{OUTDIR}/by_pac.csv"
    cat_csv = f"{OUTDIR}/sb_by_category.csv"
    
    if os.path.exists(pac_csv):
        print("ℹ️ Uploading PAC spending data...")
        pac_spending = pd.read_csv(pac_csv)
        if "pac_name" in pac_spending.columns:
            upsert_collection(pac_spending, "pac_name", "pac_spending", db, pac_csv, purge=False)
    
    if os.path.exists(cat_csv):
        print("ℹ️ Uploading expenditure categories...")
        categories = pd.read_csv(cat_csv)
        # Create composite key for categories - sanitize for Firestore
        if "pac_name" in categories.columns and "disbursement_description" in categories.columns:
            import re
            categories["category_id"] = categories.apply(
                lambda row: f"{row['pac_name']}_{re.sub(r'[^a-zA-Z0-9_-]', '_', str(row['disbursement_description'])[:50])}",
                axis=1
            )
            upsert_collection(categories, "category_id", "expenditure_categories", db, cat_csv, purge=False)
    
    # Upload categorized buckets
    bucket_csv = f"{OUTDIR}/sb_by_bucket.csv"
    if os.path.exists(bucket_csv):
        print("ℹ️ Uploading spending buckets...")
        buckets = pd.read_csv(bucket_csv)
        if "pac_name" in buckets.columns and "category_bucket" in buckets.columns:
            buckets["bucket_id"] = buckets["pac_name"] + "_" + buckets["category_bucket"].str.replace(" ", "_").str.replace("&", "and")
            upsert_collection(buckets, "bucket_id", "spending_buckets", db, bucket_csv, purge=False)
    
    # Upload PAC breakdown by candidate
    cand_pac_csv = f"{OUTDIR}/by_candidate_pac.csv"
    if os.path.exists(cand_pac_csv):
        print("ℹ️ Uploading candidate PAC breakdowns...")
        cand_pac = pd.read_csv(cand_pac_csv)
        if "candidate_id" in cand_pac.columns and "pac_name" in cand_pac.columns:
            cand_pac["breakdown_id"] = cand_pac["candidate_id"] + "_" + cand_pac["pac_name"]
            upsert_collection(cand_pac, "breakdown_id", "candidate_pac_breakdown", db, cand_pac_csv, purge=False)
    
    # Upload PAC breakdown by race
    race_pac_csv = f"{OUTDIR}/by_race_pac.csv"
    if os.path.exists(race_pac_csv):
        print("ℹ️ Uploading race PAC breakdowns...")
        race_pac = pd.read_csv(race_pac_csv)
        if "race" in race_pac.columns and "pac_name" in race_pac.columns:
            race_pac["breakdown_id"] = race_pac["race"].str.replace(":", "_") + "_" + race_pac["pac_name"]
            upsert_collection(race_pac, "breakdown_id", "race_pac_breakdown", db, race_pac_csv, purge=False)

    add_meta(db)
    print("✅ Done.")

if __name__ == "__main__":
    main()


