# process_data.py
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd

# ----- Optional ML (falls back gracefully if sklearn isn't available) -----
SKLEARN_OK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except Exception:
    SKLEARN_OK = False

# ========= Config =========
RAW_DIR = Path("data/raw")
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

ENABLE_ML = True          # turn ML on/off (will auto-skip if sklearn not present or data not sufficient)
ML_PROB_THRESHOLD = 0.80  # accept ML prediction only if probability >= threshold
VENDOR_MIN_ROWS = 5       # min labeled rows to trust a vendor's dominant channel
VENDOR_DOMINANCE = 0.70   # vendor dominant label must be >= this share

# Columns we try to keep
KEEP_COLS = [
    "sub_id",
    "expenditure_date",
    "expenditure_amount",
    "support_oppose_indicator",
    "payee_name",
    "purpose_description",
    "expenditure_description",
    "candidate_id",
    "candidate_name",
    "candidate_office",
    "candidate_office_state",
    "candidate_office_district",
    "candidate_party",
    "pac_name" # Added
]

TEXT_COLS = [
    "support_oppose_indicator",
    "payee_name",
    "purpose_description",
    "expenditure_description",
    "candidate_office",
    "candidate_office_state",
    "candidate_office_district",
    "candidate_party",
    "candidate_name",
    "pac_name"
]

# Channel rules - Expanded Taxonomy
_TAXONOMY = [
    ("Polling/Research",       r"\b(poll(?:ing)?|survey|research|focus\s*group|message\s*test|microtarget)\b"),
    ("Media Placement",        r"\b(media\s*buy|placement|buying\s*fee|reservation|insertion\s*order)\b"),
    ("Consulting/Strategy",    r"\b(consult(?:ing)?|strategy|management\s*fee|retainer|advis(ory|er)|communication\s*plan)\b"),
    ("Fundraising/Compliance", r"\b(fundrais|compliance|treasurer|report(ing)?|fec\s*filing|disbursements?)\b"),
    ("Legal/Accounting",       r"\b(legal|counsel|account(?:ing)?|audit|retainer\s*agreement)\b"),
    ("Events/Travel",          r"\b(event|venue|room\s*rent|hotel|travel|airfare|conference|catering|banquet)\b"),
    ("Production",             r"\b(production|creative|script|edit(?:ing)?|design|post[- ]?prod|graphics)\b"),
    # reinforce core channels
    ("TV",                     r"\b(tv|television|broadcast|cable|spot|reservation)\b"),
    ("Digital",                r"\b(digital|online|facebook|google|youtube|programmatic|pre-?roll|display|social)\b"),
    ("Mail",                   r"\b(mail|postage|mailer|direct\s*mail)\b"),
    ("Radio",                  r"\b(radio)\b"),
    ("Phone",                  r"\b(phone|tele ?town|robocall|sms|text)\b"),
    ("Field",                  r"\b(field|canvass|door|ground|get\s*out\s*the\s*vote|gotv)\b"),
    ("Print",                  r"\b(print|newspaper)\b"),
]
_TAXONOMY = [(lab, re.compile(pat, re.I)) for lab, pat in _TAXONOMY]

# Base rules for fallback
_BASE_RULES = [
    ("TV",         r"\b(tv|television|broadcast|cable|spot|reservation)\b"),
    ("Digital",    r"\b(digital|online|facebook|google|youtube|programmatic|pre-?roll|display|social)\b"),
    ("Mail",       r"\b(mail|postage|mailer|direct\s*mail)\b"),
    ("Radio",      r"\b(radio)\b"),
    ("Phone",      r"\b(phone|tele ?town|robocall|sms|text)\b"),
    ("Field",      r"\b(field|canvass|door|ground|get\s*out\s*the\s*vote|gotv)\b"),
    ("Print",      r"\b(print|newspaper)\b"),
    ("Production", r"\b(production|creative|edit|design|consult|strategy)\b"),
]
_BASE_RULES = [(lab, re.compile(pat, re.I)) for lab, pat in _BASE_RULES]


# ========= IO =========
def load_all_raw_data(raw_dir: Path, suffix: str = "_ie.json") -> pd.DataFrame:
    all_rows = []
    if not raw_dir.exists():
        print(f"⚠️ Raw directory not found: {raw_dir}")
        return pd.DataFrame()
        
    for f in raw_dir.glob(f"*{suffix}"):
        pac_name = f.name.replace(suffix, "")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for row in data:
                row["pac_name"] = pac_name
                all_rows.append(row)
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")
            
    return pd.DataFrame(all_rows)

# ========= Cleaning & enrichment =========
def safe_strip(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def derive_month(df: pd.DataFrame) -> None:
    # Handle both expenditure_date (IE) and disbursement_date (SB)
    col = "expenditure_date" if "expenditure_date" in df.columns else "disbursement_date"
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df["month"] = df[col].dt.to_period("M").astype(str)

def race_key(row) -> str:
    office = safe_strip(row.get("candidate_office", "")).upper()
    state  = safe_strip(row.get("candidate_office_state", "")).upper()
    dist   = safe_strip(row.get("candidate_office_district", ""))
    return f"{office}:{state}:{dist}"

def split_race(r: str) -> pd.Series:
    parts = (r or ":::").split(":")
    office = parts[0] if len(parts) > 0 else ""
    state  = parts[1] if len(parts) > 1 else ""
    dist   = parts[2] if len(parts) > 2 else ""
    office_map = {"H": "House", "S": "Senate", "P": "President"}
    return pd.Series({
        "race_office": office_map.get(office, office),
        "race_state": state,
        "race_district": dist
    })

def _blob(purpose: str, descr: str) -> str:
    return f"{(purpose or '').lower().strip()} {(descr or '').lower().strip()}".strip()

def base_channel(purpose: str, descr: str) -> str:
    text = _blob(purpose, descr)
    for label, pat in _BASE_RULES:
        if pat.search(text):
            return label
    return "Other/Unclear"

def taxonomy_channel(purpose: str, descr: str):
    text = _blob(purpose, descr)
    for label, pat in _TAXONOMY:
        if pat.search(text):
            return label, 0.95
    return "Other/Unclear", 0.0

def build_vendor_map(df: pd.DataFrame, vendor_min_rows=5, dominance=0.7):
    """
    Learn vendor -> dominant channel from rows ALREADY labeled (not Other/Unclear).
    """
    labeled = df[df["channel_rule_applied"] != "Other/Unclear"].copy()
    if labeled.empty:
        return {}
    vc = labeled.groupby("payee_name")["channel_rule_applied"].value_counts()
    vendor_map = {}
    for vendor in vc.index.get_level_values(0).unique():
        counts = vc[vendor]
        total = counts.sum()
        top = counts.idxmax()
        share = counts.max() / total if total > 0 else 0
        if total >= vendor_min_rows and share >= dominance:
            vendor_map[vendor] = top
    return vendor_map

def train_text_model(labeled_df: pd.DataFrame):
    """
    Train TF-IDF + Logistic Regression on labeled rows (channel_rule_applied).
    Returns sklearn Pipeline or None if not enough data or sklearn missing.
    """
    if not (ENABLE_ML and SKLEARN_OK):
        return None
    if labeled_df["channel_rule_applied"].nunique() < 2 or len(labeled_df) < 200:
        return None
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=3)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(labeled_df["text"], labeled_df["channel_rule_applied"])
    return pipe

def parse_race_id(race_id: str) -> dict:
    """
    Parse race ID to extract metadata.
    Format: [H/S/P]:[STATE]:[district]
    Examples:
    - H:NY:16 -> House, NY-16
    - S:CA -> Senate, CA
    - P:US -> Presidential, National
    """
    if not race_id or ":" not in race_id:
        return {}
    
    parts = race_id.split(":")
    if len(parts) < 2:
        return {}
    
    office_code = parts[0]
    office_map = {"H": "House", "S": "Senate", "P": "Presidential"}
    office = office_map.get(office_code, "Unknown")
    
    state = parts[1] if len(parts) > 1 else None
    district_num = parts[2] if len(parts) > 2 else None
    
    # Build district string
    district = None
    if office == "House" and district_num:
        district = f"{state}-{district_num}"
    elif office == "Senate":
        district = state
    elif office == "Presidential":
        district = "National"
    
    return {
        "office": office,
        "state": state,
        "district": district,
        "office_code": office_code
    }

def parse_candidate_id(cand_id: str) -> dict:
    """
    Parse FEC candidate ID to extract metadata.
    Format: [H/S/P][cycle][STATE][district][unique_id]
    Examples:
    - H4CA47085: House, CA-47
    - S8NY00082: Senate, NY
    - P80001571: Presidential
    """
    if not cand_id or len(cand_id) < 9:
        return {}
    
    office_code = cand_id[0]
    office_map = {"H": "House", "S": "Senate", "P": "Presidential"}
    office = office_map.get(office_code, "Unknown")
    
    state = cand_id[2:4]
    district_code = cand_id[4:6]
    
    # Parse district
    district = None
    if office == "House":
        try:
            district_num = int(district_code)
            district = f"{state}-{district_num:02d}" if district_num > 0 else f"{state}-AL"  # At-Large
        except:
            pass
    elif office == "Senate":
        district = state  # Senate is statewide
    elif office == "Presidential":
        district = "National"
    
    return {
        "office": office,
        "state": state,
        "district": district,
        "office_code": office_code
    }

def format_candidate_name(name: str) -> str:
    """
    Converts 'LAST, FIRST M. REP.' -> 'First M. Last'
    Removes 'REP.' suffix/prefix.
    """
    if not isinstance(name, str):
        return ""
    
    clean_name = name.replace(" REP.", "").replace(" Rep.", "").replace("REP. ", "").replace("Rep. ", "")
    clean_name = clean_name.replace("REP.", "").replace("Rep.", "").strip()
    
    if "," in clean_name:
        parts = clean_name.split(",", 1)
        last = parts[0].strip().title()
        first = parts[1].strip().title()
        return f"{first} {last}"
    
    return clean_name.title()

def categorize_operating_expenditure(description: str) -> str:
    """
    Categorizes operating expenditure descriptions into broad buckets.
    """
    if not isinstance(description, str):
        return "Other"
    
    desc_lower = description.lower()
    
    # Category rules (order matters - more specific first)
    if any(kw in desc_lower for kw in ["payroll", "salary", "salaries", "personnel", "benefits", "compensation", "wages"]):
        return "Personnel"
    elif any(kw in desc_lower for kw in ["placed media", "media placement", "media buy", "advertising", "ad buy", "broadcast"]):
        return "Media Placement"
    elif any(kw in desc_lower for kw in ["fundraising", "donor", "prospecting", "contribution"]):
        return "Fundraising"
    elif any(kw in desc_lower for kw in ["consulting", "consultant", "strategy", "advisory"]):
        return "Consulting"
    elif any(kw in desc_lower for kw in ["legal", "attorney", "compliance", "accounting", "audit"]):
        return "Legal & Compliance"
    elif any(kw in desc_lower for kw in ["rent", "office", "facility", "utilities", "lease"]):
        return "Office & Facilities"
    elif any(kw in desc_lower for kw in ["travel", "lodging", "hotel", "airfare", "transportation"]):
        return "Travel"
    elif any(kw in desc_lower for kw in ["donation", "contribution to", "transfer"]):
        return "Donations & Transfers"
    elif any(kw in desc_lower for kw in ["phone", "telecommunications", "internet", "software", "technology"]):
        return "Technology & Communications"
    elif any(kw in desc_lower for kw in ["polling", "research", "survey"]):
        return "Research & Polling"
    else:
        return "Other"

# ========= DRY aggregations =========
def summarize_with_refunds(
    df: pd.DataFrame,
    group_cols: List[str],
    amount_col: str = "expenditure_amount",
    id_col: str = "sub_id",
) -> pd.DataFrame:
    base = (
        df.groupby(group_cols, as_index=False)
          .agg(net_amount=(amount_col, "sum"), lines=(id_col, "count"))
    )
    pos = (
        df[df[amount_col] > 0]
        .groupby(group_cols, as_index=False)
        .agg(gross_positive_amount=(amount_col, "sum"),
             positive_lines=(id_col, "count"))
    )
    neg = (
        df[df[amount_col] < 0]
        .groupby(group_cols, as_index=False)
        .agg(negative_refunds_amount=(amount_col, "sum"),
             negative_lines=(id_col, "count"))
    )
    out = (
        base.merge(pos, on=group_cols, how="left")
            .merge(neg, on=group_cols, how="left")
            .fillna({
                "gross_positive_amount": 0.0,
                "positive_lines": 0,
                "negative_refunds_amount": 0.0,
                "negative_lines": 0
            })
    )
    out["refund_ratio_vs_gross"] = out.apply(
        lambda r: (abs(r["negative_refunds_amount"]) / r["gross_positive_amount"])
        if r["gross_positive_amount"] > 0 else 0.0,
        axis=1
    )
    return out

def support_oppose_pivot(
    df: pd.DataFrame,
    index_cols: List[str],
    amount_col: str = "expenditure_amount",
    id_col: str = "sub_id",
    so_col: str = "support_oppose_indicator",
) -> pd.DataFrame:
    tmp = df[df[so_col].isin(["S", "O"])].copy()
    grp = (
        tmp.groupby(index_cols + [so_col], as_index=False)
           .agg(amount=(amount_col, "sum"), lines=(id_col, "count"))
    )
    wide_amt = (
        grp.pivot(index=index_cols, columns=so_col, values="amount")
           .rename(columns={"S": "support_amount", "O": "oppose_amount"})
           .fillna(0.0)
    )
    wide_lines = (
        grp.pivot(index=index_cols, columns=so_col, values="lines")
           .rename(columns={"S": "support_lines", "O": "oppose_lines"})
           .fillna(0)
    )
    out = wide_amt.join(wide_lines, how="outer").fillna(0)
    out["total_ie_amount"] = out["support_amount"] + out["oppose_amount"]
    out["total_lines"] = out["support_lines"] + out["oppose_lines"]
    out["net_support_amount"] = out["support_amount"] - out["oppose_amount"]
    out["support_ratio"] = out.apply(
        lambda r: (r["support_amount"] / r["total_ie_amount"]) if r["total_ie_amount"] > 0 else 0.0,
        axis=1
    )
    out["oppose_ratio"] = out.apply(
        lambda r: (r["oppose_amount"] / r["total_ie_amount"]) if r["total_ie_amount"] > 0 else 0.0,
        axis=1
    )
    return out.reset_index()

def compute_channel_mix(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Computes the percentage of spending per channel for each group.
    Returns a DataFrame with group_cols and pct_{channel} columns.
    """
    # Filter to positive spending only for mix calculation
    pos = df[df["expenditure_amount"] > 0].copy()
    
    # Group by entity + channel
    grouped = pos.groupby(group_cols + ["channel"])["expenditure_amount"].sum().reset_index()
    
    # Pivot to wide format
    pivoted = grouped.pivot(index=group_cols, columns="channel", values="expenditure_amount").fillna(0.0)
    
    # Calculate total per entity
    pivoted["_total"] = pivoted.sum(axis=1)
    
    # Calculate percentages
    out = pd.DataFrame(index=pivoted.index)
    for col in pivoted.columns:
        if col == "_total": continue
        # Create clean column name (e.g. "pct_media_placement")
        clean_col = "pct_" + re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        out[clean_col] = pivoted[col] / pivoted["_total"]
        
    return out.reset_index()

# ========= Main =========
def consolidate_candidate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies candidates with multiple IDs and unifies them under a single canonical ID
    based on the highest absolute spending.
    """
    print("ℹ️ Consolidating candidate IDs...")
    
    # Group by name and ID, sum spending
    temp_stats = df.groupby(["candidate_name", "candidate_id"])["expenditure_amount"].sum().reset_index()
    
    # For each name, pick the ID with max absolute spending
    temp_stats["abs_amount"] = temp_stats["expenditure_amount"].abs()
    canonical = temp_stats.sort_values("abs_amount", ascending=False).drop_duplicates("candidate_name")
    
    # Create map: name -> canonical_id
    name_to_id = dict(zip(canonical["candidate_name"], canonical["candidate_id"]))
    
    # Update df with canonical IDs
    df["candidate_id"] = df["candidate_name"].map(name_to_id).fillna(df["candidate_id"])
    
    print(f"ℹ️ Consolidated IDs for {len(name_to_id)} candidates.")
    return df

def main():
    print("🚀 Loading raw IE data...")
    df_ie = load_all_raw_data(RAW_DIR, "_ie.json")
    
    print("🚀 Loading raw SB (Operating) data...")
    df_sb = load_all_raw_data(RAW_DIR, "_sb.json")

    # --- Process IEs ---
    if not df_ie.empty:
        print("ℹ️ Processing IEs...")
        df_ie = ensure_columns(df_ie, KEEP_COLS)
        if "sub_id" in df_ie.columns:
            df_ie = df_ie.drop_duplicates(subset=["sub_id"])
        
        for c in TEXT_COLS:
            if c in df_ie.columns:
                df_ie[c] = df_ie[c].apply(safe_strip)
        
        df_ie["candidate_name"] = df_ie["candidate_name"].apply(format_candidate_name)
        df_ie["expenditure_amount"] = pd.to_numeric(df_ie["expenditure_amount"], errors="coerce").fillna(0.0)
        derive_month(df_ie)
        df_ie["race"] = df_ie.apply(race_key, axis=1)
        
        # --- Advanced Channel Classification ---
        print("ℹ️ Running advanced channel classification...")
        
        # 0) Start point: compute base channel
        df_ie["channel_base"] = df_ie.apply(lambda r: base_channel(r.get("purpose_description",""), r.get("expenditure_description","")), axis=1)

        # 1) Rules/taxonomy pass
        tax = df_ie.apply(lambda r: taxonomy_channel(r.get("purpose_description",""), r.get("expenditure_description","")), axis=1)
        df_ie["channel_rule_applied"] = [lab for (lab, _) in tax]
        
        # seed refined channel: keep base if not Other/Unclear; else take taxonomy
        df_ie["channel_refined"] = df_ie["channel_base"]
        take_tax = df_ie["channel_refined"].eq("Other/Unclear") & df_ie["channel_rule_applied"].ne("Other/Unclear")
        df_ie.loc[take_tax, "channel_refined"] = df_ie.loc[take_tax, "channel_rule_applied"]
        
        # 2) Vendor backfill on rows still Other/Unclear
        still_unclear = df_ie["channel_refined"].eq("Other/Unclear")
        vendor_map = build_vendor_map(df_ie, VENDOR_MIN_ROWS, VENDOR_DOMINANCE)
        if vendor_map and still_unclear.any():
            print(f"ℹ️ Vendor backfill: mapped {len(vendor_map)} vendors.")
            m = df_ie.loc[still_unclear, "payee_name"].map(vendor_map)
            hit = m.notna()
            df_ie.loc[still_unclear & hit, "channel_refined"] = m[hit]
            
        # 3) Optional ML on remaining Other/Unclear
        still_unclear = df_ie["channel_refined"].eq("Other/Unclear")
        if ENABLE_ML and SKLEARN_OK and still_unclear.any():
            print("ℹ️ Training ML model for remaining unclear items...")
            labeled = df_ie[df_ie["channel_refined"] != "Other/Unclear"].copy()
            labeled["text"] = (labeled["purpose_description"].fillna("") + " " + labeled["expenditure_description"].fillna("")).str.lower()
            # Use refined labels as ground truth for training
            labeled["channel_rule_applied"] = labeled["channel_refined"] 
            
            pipe = train_text_model(labeled)
            if pipe is not None:
                texts = (df_ie.loc[still_unclear, "purpose_description"].fillna("") + " " +
                         df_ie.loc[still_unclear, "expenditure_description"].fillna("")).str.lower().tolist()
                probs = pipe.predict_proba(texts)
                labels = pipe.classes_
                idxmax = probs.argmax(axis=1)
                pred_labels = [labels[i] for i in idxmax]
                pred_probs  = [float(probs[j, idxmax[j]]) for j in range(len(idxmax))]
                
                accept = [p >= ML_PROB_THRESHOLD for p in pred_probs]
                accept_idx = df_ie.loc[still_unclear].index[accept]
                
                if len(accept_idx) > 0:
                    print(f"ℹ️ ML classified {len(accept_idx)} items.")
                    df_ie.loc[accept_idx, "channel_refined"] = [pred_labels[k] for k, a in enumerate(accept) if a]
            else:
                print("ℹ️ ML skipped (not enough labeled diversity).")
        elif ENABLE_ML and not SKLEARN_OK:
            print("ℹ️ scikit-learn not installed; ML step skipped.")

        # Final channel assignment
        df_ie["channel"] = df_ie["channel_refined"]
        
        race_parts = df_ie["race"].apply(split_race)
        df_ie = pd.concat([df_ie, race_parts], axis=1)

        # Consolidate IDs to ensure linking consistency
        df_ie = consolidate_candidate_ids(df_ie)
        
        # Parse candidate IDs to extract metadata
        print("ℹ️ Parsing candidate ID metadata...")
        id_metadata = df_ie["candidate_id"].apply(parse_candidate_id)
        df_ie["office"] = id_metadata.apply(lambda x: x.get("office"))
        df_ie["state_code"] = id_metadata.apply(lambda x: x.get("state"))
        df_ie["district"] = id_metadata.apply(lambda x: x.get("district"))

        # Aggregates
        by_race = summarize_with_refunds(df_ie, ["race"])
        by_race_net = support_oppose_pivot(df_ie, ["race"])
        race_cands = df_ie.groupby("race")["candidate_name"].unique().apply(lambda x: sorted(list(x))).reset_index()
        race_cands = race_cands.rename(columns={"candidate_name": "candidates"})
        by_race = by_race.merge(by_race_net, on="race", how="left")
        by_race = by_race.merge(race_cands, on="race", how="left")
        
        # Parse race IDs to extract metadata
        print("ℹ️ Parsing race ID metadata...")
        race_metadata = by_race["race"].apply(parse_race_id)
        by_race["office"] = race_metadata.apply(lambda x: x.get("office"))
        by_race["state"] = race_metadata.apply(lambda x: x.get("state"))
        by_race["district"] = race_metadata.apply(lambda x: x.get("district"))
        by_race = by_race.sort_values("total_ie_amount", ascending=False)

        by_candidate = summarize_with_refunds(df_ie, ["candidate_id", "candidate_name"])
        by_candidate_net = support_oppose_pivot(df_ie, ["candidate_id", "candidate_name"])
        by_candidate = by_candidate.merge(by_candidate_net, on=["candidate_id", "candidate_name"], how="left")
        
        # Add metadata from parsed IDs (take first occurrence per candidate)
        cand_metadata = df_ie[["candidate_id", "office", "state_code", "district"]].drop_duplicates("candidate_id")
        by_candidate = by_candidate.merge(cand_metadata, on="candidate_id", how="left")
        
        # Calculate Channel Mix for Candidates
        print("ℹ️ Computing candidate channel mix...")
        cand_mix = compute_channel_mix(df_ie, ["candidate_id"])
        by_candidate = by_candidate.merge(cand_mix, on="candidate_id", how="left")
        
        by_candidate = by_candidate.sort_values("total_ie_amount", ascending=False)

        by_pac = summarize_with_refunds(df_ie, ["pac_name"])
        by_pac = by_pac.sort_values("net_amount", ascending=False)

        # NEW: PAC-by-PAC breakdowns
        # By Race and PAC
        by_race_pac = df_ie.groupby(["race", "pac_name"], as_index=False).agg({
            "expenditure_amount": "sum",
            "sub_id": "count",
            "channel": lambda x: x.value_counts().to_dict()
        })
        by_race_pac = by_race_pac.rename(columns={"sub_id": "lines", "channel": "channel_breakdown"})
        
        # By Candidate and PAC  
        by_cand_pac = df_ie.groupby(["candidate_id", "candidate_name", "pac_name"], as_index=False).agg({
            "expenditure_amount": "sum",
            "sub_id": "count"
        })
        by_cand_pac = by_cand_pac.rename(columns={"sub_id": "lines"})

        print(f"💾 Saving IE processed files to {OUTDIR}...")
        df_ie.to_csv(OUTDIR / "all_ie_enriched.csv", index=False)
        by_race.to_csv(OUTDIR / "by_race.csv", index=False)
        by_candidate.to_csv(OUTDIR / "by_candidate.csv", index=False)
        by_pac.to_csv(OUTDIR / "by_pac.csv", index=False)
        by_race_pac.to_csv(OUTDIR / "by_race_pac.csv", index=False)
        by_cand_pac.to_csv(OUTDIR / "by_candidate_pac.csv", index=False)

    # --- Process SBs ---
    if not df_sb.empty:
        print("ℹ️ Processing SBs...")
        # Basic cleaning for SB
        amt_col = "disbursement_amount" if "disbursement_amount" in df_sb.columns else "expenditure_amount"
        df_sb["expenditure_amount"] = pd.to_numeric(df_sb[amt_col], errors="coerce").fillna(0.0)
        derive_month(df_sb)
        
        # Add category buckets
        df_sb["category_bucket"] = df_sb["disbursement_description"].apply(categorize_operating_expenditure)
        
        # Aggregate by Category (disbursement_description)
        sb_by_cat = df_sb.groupby(["pac_name", "disbursement_description"], as_index=False)["expenditure_amount"].sum()
        sb_by_cat["category_bucket"] = sb_by_cat["disbursement_description"].apply(categorize_operating_expenditure)
        sb_by_cat = sb_by_cat.sort_values("expenditure_amount", ascending=False)
        
        # Aggregate by Category Bucket
        sb_by_bucket = df_sb.groupby(["pac_name", "category_bucket"], as_index=False).agg({
            "expenditure_amount": "sum",
            "disbursement_description": "count"
        })
        sb_by_bucket = sb_by_bucket.rename(columns={"disbursement_description": "line_count"})
        sb_by_bucket = sb_by_bucket.sort_values("expenditure_amount", ascending=False)
        
        # Aggregate by Recipient
        sb_by_recipient = df_sb.groupby(["pac_name", "recipient_name"], as_index=False)["expenditure_amount"].sum()
        sb_by_recipient = sb_by_recipient.sort_values("expenditure_amount", ascending=False)

        print(f"💾 Saving SB processed files to {OUTDIR}...")
        df_sb.to_csv(OUTDIR / "all_sb_enriched.csv", index=False)
        sb_by_cat.to_csv(OUTDIR / "sb_by_category.csv", index=False)
        sb_by_bucket.to_csv(OUTDIR / "sb_by_bucket.csv", index=False)
        sb_by_recipient.to_csv(OUTDIR / "sb_by_recipient.csv", index=False)

    print("✅ Done.")

if __name__ == "__main__":
    main()
