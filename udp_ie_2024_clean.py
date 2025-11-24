# categorize_udp_ie.py
# UDP IE processing: DRY refactor + channel bucketing + refund breakdown
# + candidate & race net-support, and standard aggregates

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd


# ========= Config =========
INPUT = "/Users/neilkhandelwal/Documents/PAC_TRACK/udp_ie_2024.json"  # or .jsonl
OUTDIR = Path("udp_ie_out")
OUTDIR.mkdir(exist_ok=True)

# Columns we try to keep; missing ones will be created empty
KEEP_COLS = [
    "sub_id",
    "expenditure_date",
    "expenditure_amount",
    "support_oppose_indicator",   # 'S' or 'O'
    "payee_name",
    "purpose_description",
    "expenditure_description",
    "candidate_id",
    "candidate_name",
    "candidate_office",           # 'H','S','P'
    "candidate_office_state",
    "candidate_office_district",
    "candidate_party",
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
]

# Channel rules (precompiled)
_CHANNEL_RULES_RAW = [
    ("TV",         r"\b(tv|television|broadcast|cable|spot|reservation)\b"),
    ("Digital",    r"\b(digital|online|facebook|google|youtube|programmatic|pre-?roll|display)\b"),
    ("Mail",       r"\b(mail|postage|mailer|direct mail)\b"),
    ("Radio",      r"\b(radio)\b"),
    ("Phone",      r"\b(phone|tele ?town|robocall|sms|text)\b"),
    ("Field",      r"\b(field|canvass|door|ground|get out the vote|gotv)\b"),
    ("Print",      r"\b(print|newspaper)\b"),
    ("Production", r"\b(production|creative|edit|design|consult|strategy)\b"),
]
_CHANNEL_RULES = [(lab, re.compile(pat, re.I)) for lab, pat in _CHANNEL_RULES_RAW]


# ========= IO =========
def load_ie(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p.resolve()}")
    if path.endswith(".jsonl"):
        rows = [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


# ========= Cleaning & enrichment =========
def safe_strip(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""  # create empty text col; amounts handled separately
    return df[cols].copy()

def derive_month(df: pd.DataFrame) -> None:
    df["expenditure_date"] = pd.to_datetime(df["expenditure_date"], errors="coerce")
    df["month"] = df["expenditure_date"].dt.to_period("M").astype(str)

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

def categorize_channel_dual(purpose: str, descr: str) -> str:
    blob = f"{(purpose or '').lower()} {(descr or '').lower()}".strip()
    for label, pattern in _CHANNEL_RULES:
        if pattern.search(blob):
            return label
    return "Other/Unclear"


# ========= DRY aggregations =========
def summarize_with_refunds(
    df: pd.DataFrame,
    group_cols: List[str],
    amount_col: str = "expenditure_amount",
    id_col: str = "sub_id",
) -> pd.DataFrame:
    """
    Returns one row per group with:
      - net_amount
      - lines
      - gross_positive_amount, positive_lines
      - negative_refunds_amount, negative_lines
    """
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
    """Wide table with support/oppose amounts & lines, plus totals/ratios/sentiment."""
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
    def _sentiment(r):
        if r["total_ie_amount"] <= 0:
            return "None"
        if r["support_ratio"] >= 0.60:
            return "Support-lean"
        if r["support_ratio"] <= 0.40:
            return "Oppose-lean"
        return "Mixed"
    out["dominant_ie_sentiment"] = out.apply(_sentiment, axis=1)
    return out.reset_index()


# ========= Main =========
def main():
    # ----- load & ensure schema -----
    df = load_ie(INPUT)
    df = ensure_columns(df, KEEP_COLS)

    # drop dups on sub_id if present
    if "sub_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["sub_id"])
        dropped = before - len(df)
        if dropped:
            print(f"ℹ️ Dropped {dropped} duplicate rows by sub_id")

    # normalize text
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].apply(safe_strip)

    # amounts numeric
    df["expenditure_amount"] = pd.to_numeric(df["expenditure_amount"], errors="coerce").fillna(0.0)

    # month, race, channel
    derive_month(df)
    df["race"] = df.apply(race_key, axis=1)
    df["channel"] = df.apply(
        lambda r: categorize_channel_dual(r.get("purpose_description", ""), r.get("expenditure_description", "")),
        axis=1
    )
    race_parts = df["race"].apply(split_race)
    df = pd.concat([df, race_parts], axis=1)

    # ----- core aggregates (DRY) -----
    # by race (refund-aware)
    by_race = summarize_with_refunds(df, ["race"])
    by_race = by_race.sort_values("net_amount", ascending=False)

    # by candidate (refund-aware)
    by_candidate = summarize_with_refunds(df, ["candidate_id", "candidate_name"])
    by_candidate = by_candidate.sort_values("net_amount", ascending=False)

    # by state (refund-aware)
    by_state = summarize_with_refunds(df, ["candidate_office_state"])
    by_state = by_state.sort_values("net_amount", ascending=False)

    # by payee (refund-aware)
    by_payee = summarize_with_refunds(df, ["payee_name"])
    by_payee = by_payee.sort_values("net_amount", ascending=False)

    # channels overall
    by_channel = summarize_with_refunds(df, ["channel"])
    by_channel = by_channel.sort_values("net_amount", ascending=False)

    # channel mix within race (refund-aware)
    by_race_channel = summarize_with_refunds(df, ["race", "channel"])
    by_race_channel = by_race_channel.sort_values(["race", "net_amount"], ascending=[True, False])

    # monthly trend overall (refund-aware)
    by_month = summarize_with_refunds(df, ["month"])
    by_month = by_month.sort_values("month")

    # ----- support vs oppose wide tables -----
    # candidate-level net support
    by_candidate_net = support_oppose_pivot(df, ["candidate_id", "candidate_name"])
    by_candidate_net = by_candidate_net.sort_values(["net_support_amount", "total_ie_amount"], ascending=[False, False])

    # race-level net support
    by_race_net = support_oppose_pivot(df, ["race"])
    by_race_net = by_race_net.sort_values(["net_support_amount", "total_ie_amount"], ascending=[False, False])

    # ----- QA report -----
    qa_rows = []
    nat_count = df["expenditure_date"].isna().sum()
    qa_rows.append(("date_parse_failures_NaT", int(nat_count)))
    neg_ct = (df["expenditure_amount"] < 0).sum()
    qa_rows.append(("negative_amount_rows", int(neg_ct)))
    total_amt = df["expenditure_amount"].sum()
    other_amt = df.loc[df["channel"] == "Other/Unclear", "expenditure_amount"].sum()
    other_share = float(other_amt / total_amt) if total_amt else 0.0
    qa_rows.append(("other_unclear_channel_amount", float(other_amt)))
    qa_rows.append(("other_unclear_channel_share", round(other_share, 4)))
    empty_race_ct = (df["race"] == "::").sum()
    qa_rows.append(("empty_race_key_rows", int(empty_race_ct)))
    qa = pd.DataFrame(qa_rows, columns=["metric", "value"])

    # ----- save -----
    df.to_csv(OUTDIR / "udp_ie_raw_clean_enriched.csv", index=False)
    by_race.to_csv(OUTDIR / "udp_ie_by_race.csv", index=False)
    by_candidate.to_csv(OUTDIR / "udp_ie_by_candidate.csv", index=False)
    by_state.to_csv(OUTDIR / "udp_ie_by_state.csv", index=False)
    by_payee.to_csv(OUTDIR / "udp_ie_top_payees.csv", index=False)
    by_channel.to_csv(OUTDIR / "udp_ie_by_channel.csv", index=False)
    by_race_channel.to_csv(OUTDIR / "udp_ie_by_race_channel.csv", index=False)
    by_month.to_csv(OUTDIR / "udp_ie_by_month.csv", index=False)
    by_candidate_net.to_csv(OUTDIR / "udp_ie_candidate_net_support.csv", index=False)
    by_race_net.to_csv(OUTDIR / "udp_ie_race_net_support.csv", index=False)
    qa.to_csv(OUTDIR / "udp_ie_qa_report.csv", index=False)

    print("✅ Wrote outputs to:", OUTDIR.resolve())
    for f in [
        "udp_ie_raw_clean_enriched.csv",
        "udp_ie_by_race.csv",
        "udp_ie_by_candidate.csv",
        "udp_ie_by_state.csv",
        "udp_ie_top_payees.csv",
        "udp_ie_by_channel.csv",
        "udp_ie_by_race_channel.csv",
        "udp_ie_by_month.csv",
        "udp_ie_candidate_net_support.csv",
        "udp_ie_race_net_support.csv",
        "udp_ie_qa_report.csv",
    ]:
        print("   -", f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        INPUT = sys.argv[1]
    main()
