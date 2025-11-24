# T10Metrics.py
# Core metrics (candidate & race) with integrated channel refinement:
# rules -> vendor backfill -> optional ML (TF-IDF + Logistic Regression)

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import re

# ----- Optional ML (falls back gracefully if sklearn isn't available) -----
SKLEARN_OK = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except Exception:
    SKLEARN_OK = False

# ===================== Config =====================
INPATH = Path("udp_ie_out/udp_ie_raw_clean_enriched.csv")  # input from your cleaning pipeline
OUTDIR = Path("udp_ie_out")
OUTDIR.mkdir(exist_ok=True)

LAST_UPDATED = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

ENABLE_ML = True          # turn ML on/off (will auto-skip if sklearn not present or data not sufficient)
ML_PROB_THRESHOLD = 0.80  # accept ML prediction only if probability >= threshold
VENDOR_MIN_ROWS = 5       # min labeled rows to trust a vendor's dominant channel
VENDOR_DOMINANCE = 0.70   # vendor dominant label must be >= this share

# ===================== Channel rules =====================
# Base quick rules (kept simple; taxonomy below is richer)
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

# Expanded taxonomy used in the refinement step (order matters, first match wins)
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

# ===================== Core metric helpers =====================
def nz(x): return 0.0 if pd.isna(x) else float(x)

def compute_support_oppose(df):
    sup = df.loc[df["support_oppose_indicator"] == "S", "expenditure_amount"].sum()
    opp = df.loc[df["support_oppose_indicator"] == "O", "expenditure_amount"].sum()
    total = nz(sup) + nz(opp)
    return pd.Series({
        "support_amount": nz(sup),
        "oppose_amount": nz(opp),
        "net_support_amount": nz(sup) - nz(opp),
        "support_ratio": (float(sup) / float(total)) if total != 0 else 0.0
    })

def compute_refund_metrics(df):
    gross_pos = df.loc[df["expenditure_amount"] > 0, "expenditure_amount"].sum()
    neg_ref  = df.loc[df["expenditure_amount"] < 0, "expenditure_amount"].sum()
    pos_lines = int((df["expenditure_amount"] > 0).sum())
    neg_lines = int((df["expenditure_amount"] < 0).sum())
    refund_ratio = (abs(neg_ref) / gross_pos) if gross_pos > 0 else 0.0
    return pd.Series({
        "gross_positive_amount": nz(gross_pos),
        "negative_refunds_amount": nz(neg_ref),
        "refund_ratio_vs_gross": refund_ratio,
        "positive_lines": pos_lines,
        "negative_lines": neg_lines
    })

def compute_channel_mix(df, channel_col="channel_final", amt_col="expenditure_amount"):
    """
    Channel mix as % of gross-positive spend by channel (refund-insensitive).
    Fallback to net totals if no positives exist.
    """
    pos = df[df[amt_col] > 0]
    if len(pos) == 0:
        pos = df
    def slug(s): return str(s).lower().replace("/", "_").replace(" ", "_")
    totals = pos.groupby(channel_col, dropna=False)[amt_col].sum()
    grand = float(totals.sum()) if len(totals) else 0.0
    mix = {}
    if grand > 0:
        for ch, val in totals.items():
            ch_name = "unknown" if pd.isna(ch) or ch == "" else slug(ch)
            mix[f"pct_{ch_name}"] = float(val) / grand
    return mix

def compute_committee_influence(df):
    """
    Top committee and concentration among top 3 committees, based on gross-positive spend.
    Uses committee_name if present, else committee_id. Returns blanks if unavailable.
    """
    committee_key = None
    for k in ["committee_name", "committee_id"]:
        if k in df.columns:
            committee_key = k
            break
    if committee_key is None:
        return pd.Series({"top_committee": "", "committee_concentration_pct": 0.0})
    pos = df[df["expenditure_amount"] > 0]
    if len(pos) == 0:
        return pd.Series({"top_committee": "", "committee_concentration_pct": 0.0})
    sums = (pos.groupby(committee_key)["expenditure_amount"].sum().sort_values(ascending=False))
    total = float(sums.sum())
    if total <= 0:
        return pd.Series({"top_committee": "", "committee_concentration_pct": 0.0})
    top_committee = str(sums.index[0])
    top3_share = float(sums.iloc[:3].sum()) / total if len(sums) > 0 else 0.0
    return pd.Series({"top_committee": top_committee, "committee_concentration_pct": top3_share})

def finalize(df, id_cols):
    # map internal aggregations into final names & order columns
    if "expenditure_amount_sum" in df.columns:
        df["total_ie_amount"] = df.pop("expenditure_amount_sum")
    if "lines_count" in df.columns:
        df["lines"] = df.pop("lines_count")

    preferred = id_cols + [
        "total_ie_amount",
        "support_amount",
        "oppose_amount",
        "net_support_amount",
        "support_ratio",
        "refund_ratio_vs_gross",
        "gross_positive_amount",
        "negative_refunds_amount",
        "positive_lines",
        "negative_lines",
        "top_committee",
        "committee_concentration_pct",
    ]
    channel_cols = [c for c in df.columns if c.startswith("pct_")]
    preferred += channel_cols
    cols = [c for c in preferred if c in df.columns]
    df["last_updated"] = LAST_UPDATED
    cols += ["last_updated"]
    return df[cols]

# ===================== Load =====================
if not INPATH.exists():
    raise FileNotFoundError(f"Expected input at {INPATH.resolve()} — run your cleaning script first.")

raw = pd.read_csv(INPATH, dtype=str)
raw["expenditure_amount"] = pd.to_numeric(raw.get("expenditure_amount", 0.0), errors="coerce").fillna(0.0)
raw["support_oppose_indicator"] = raw.get("support_oppose_indicator", "").str.strip().str.upper()

# Ensure text cols exist
for c in ["purpose_description", "expenditure_description", "payee_name", "channel",
          "committee_id", "committee_name", "candidate_id", "candidate_name", "race"]:
    if c not in raw.columns:
        raw[c] = ""

# ===================== Channel Refinement =====================
# 0) Start point: use existing 'channel' if present; otherwise compute a base channel
if "channel" in raw.columns and raw["channel"].notna().any():
    raw["channel_base"] = raw["channel"].fillna("Other/Unclear").replace("", "Other/Unclear")
else:
    raw["channel_base"] = raw.apply(lambda r: base_channel(r.get("purpose_description",""), r.get("expenditure_description","")), axis=1)

# 1) Rules/taxonomy pass
tax = raw.apply(lambda r: taxonomy_channel(r.get("purpose_description",""), r.get("expenditure_description","")), axis=1)
raw["channel_rule_applied"] = [lab for (lab, _) in tax]
raw["channel_rule_conf"] = [conf for (_, conf) in tax]

# seed refined channel: keep base if not Other/Unclear; else take taxonomy
raw["channel_refined"] = raw["channel_base"]
take_tax = raw["channel_refined"].eq("Other/Unclear") & raw["channel_rule_applied"].ne("Other/Unclear")
raw.loc[take_tax, "channel_refined"] = raw.loc[take_tax, "channel_rule_applied"]
raw["channel_confidence"] = raw["channel_refined"].apply(lambda x: 1.0 if x != "Other/Unclear" else 0.0)
raw["channel_source"] = raw["channel_refined"].apply(lambda x: "base" if x != "Other/Unclear" else "unclear")
raw.loc[take_tax, "channel_confidence"] = 0.95
raw.loc[take_tax, "channel_source"] = "rules"

# 2) Vendor backfill on rows still Other/Unclear
still_unclear = raw["channel_refined"].eq("Other/Unclear")
vendor_map = build_vendor_map(raw, VENDOR_MIN_ROWS, VENDOR_DOMINANCE)
if vendor_map and still_unclear.any():
    m = raw.loc[still_unclear, "payee_name"].map(vendor_map)
    hit = m.notna()
    raw.loc[still_unclear & hit, "channel_refined"] = m[hit]
    raw.loc[still_unclear & hit, "channel_confidence"] = 0.90
    raw.loc[still_unclear & hit, "channel_source"] = "vendor_map"

# 3) Optional ML on remaining Other/Unclear
still_unclear = raw["channel_refined"].eq("Other/Unclear")
if ENABLE_ML and SKLEARN_OK and still_unclear.any():
    labeled = raw[raw["channel_refined"] != "Other/Unclear"].copy()
    labeled["text"] = (labeled["purpose_description"].fillna("") + " " + labeled["expenditure_description"].fillna("")).str.lower()
    labeled = labeled.rename(columns={"channel_refined": "channel_rule_applied"})
    pipe = train_text_model(labeled)
    if pipe is not None:
        texts = (raw.loc[still_unclear, "purpose_description"].fillna("") + " " +
                 raw.loc[still_unclear, "expenditure_description"].fillna("")).str.lower().tolist()
        probs = pipe.predict_proba(texts)
        labels = pipe.classes_
        idxmax = probs.argmax(axis=1)
        pred_labels = [labels[i] for i in idxmax]
        pred_probs  = [float(probs[j, idxmax[j]]) for j in range(len(idxmax))]
        accept = [p >= ML_PROB_THRESHOLD for p in pred_probs]
        accept_idx = raw.loc[still_unclear].index[accept]
        if len(accept_idx) > 0:
            raw.loc[accept_idx, "channel_refined"] = [pred_labels[k] for k, a in enumerate(accept) if a]
            raw.loc[accept_idx, "channel_confidence"] = [pred_probs[k] for k, a in enumerate(accept) if a]
            raw.loc[accept_idx, "channel_source"] = "ml"
    else:
        print("ℹ️ ML skipped (not enough labeled diversity).")
elif ENABLE_ML and not SKLEARN_OK:
    print("ℹ️ scikit-learn not installed; ML step skipped.")

# Final channel field used for metrics
raw["channel_final"] = raw["channel_refined"]

# (Optional) write a tiny refinement report
base_other = (raw["channel_base"] == "Other/Unclear").sum()
final_other = (raw["channel_final"] == "Other/Unclear").sum()
pd.DataFrame(
    [("base_other_rows", int(base_other)),
     ("final_other_rows", int(final_other)),
     ("delta_other_rows", int(base_other - final_other))]
    , columns=["metric","value"]
).to_csv(OUTDIR / "core_channel_refine_summary.csv", index=False)

# ===================== Build Core Metrics (CANDIDATE) =====================
cand_keys = ["candidate_id", "candidate_name"]
for c in cand_keys:
    if c not in raw.columns:
        raw[c] = ""

cand_group = raw.groupby(cand_keys, dropna=False)

cand_agg = pd.DataFrame({
    "expenditure_amount_sum": cand_group["expenditure_amount"].sum(),
    "lines_count": cand_group.size()
}).reset_index()

cand_support_oppose = cand_group.apply(compute_support_oppose, include_groups=False).reset_index()
cand_refunds        = cand_group.apply(compute_refund_metrics, include_groups=False).reset_index()
cand_committees     = cand_group.apply(compute_committee_influence, include_groups=False).reset_index()

# Channel mix based on refined channel
cand_mix_records = []
for keys, df_sub in cand_group:
    if isinstance(keys, tuple):
        rec = dict(zip(cand_keys, keys))
    else:
        rec = {cand_keys[0]: keys, cand_keys[1]: ""}
    rec.update(compute_channel_mix(df_sub))  # uses channel_final internally
    cand_mix_records.append(rec)
cand_mix = pd.DataFrame(cand_mix_records)

cand_final = (cand_agg
              .merge(cand_support_oppose, on=cand_keys, how="left")
              .merge(cand_refunds, on=cand_keys, how="left")
              .merge(cand_mix, on=cand_keys, how="left")
              .merge(cand_committees, on=cand_keys, how="left"))

cand_final = finalize(cand_final, cand_keys)
cand_final.to_csv(OUTDIR / "core_metrics_by_candidate.csv", index=False)

# ===================== Build Core Metrics (RACE) =====================
# Ensure race exists (H:NY:10, etc.)
if "race" not in raw.columns:
    office = raw.get("candidate_office", "").fillna("").str.upper()
    state  = raw.get("candidate_office_state", "").fillna("").str.upper()
    dist   = raw.get("candidate_office_district", "").fillna("")
    raw["race"] = office + ":" + state + ":" + dist

race_keys = ["race"]
race_group = raw.groupby(race_keys, dropna=False)

race_agg = pd.DataFrame({
    "expenditure_amount_sum": race_group["expenditure_amount"].sum(),
    "lines_count": race_group.size()
}).reset_index()

race_support_oppose = race_group.apply(compute_support_oppose, include_groups=False).reset_index()
race_refunds        = race_group.apply(compute_refund_metrics, include_groups=False).reset_index()
race_committees     = race_group.apply(compute_committee_influence, include_groups=False).reset_index()

race_mix_records = []
for keys, df_sub in race_group:
    race_val = keys if not isinstance(keys, tuple) else keys[0]
    rec = {"race": race_val}
    rec.update(compute_channel_mix(df_sub))  # uses channel_final internally
    race_mix_records.append(rec)
race_mix = pd.DataFrame(race_mix_records)

race_final = (race_agg
              .merge(race_support_oppose, on=race_keys, how="left")
              .merge(race_refunds, on=race_keys, how="left")
              .merge(race_mix, on=race_keys, how="left")
              .merge(race_committees, on=race_keys, how="left"))

race_final = finalize(race_final, race_keys)
# Optionally rename total to cost_of_contest at race-level:
race_final.rename(columns={"total_ie_amount": "cost_of_contest"}, inplace=True)

race_final.to_csv(OUTDIR / "core_metrics_by_race.csv", index=False)

print("✅ Wrote:")
print("  -", (OUTDIR / "core_metrics_by_candidate.csv").resolve())
print("  -", (OUTDIR / "core_metrics_by_race.csv").resolve())
print("  -", (OUTDIR / "core_channel_refine_summary.csv").resolve())
