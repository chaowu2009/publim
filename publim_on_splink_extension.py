"""
PUBLIM-on-Splink Extension
==========================

Purpose
-------
Use Splink as the candidate-generation / comparison engine, then apply a
PUBLIM-style uniqueness score on top of Splink prediction pairs.

This gives you:

1. Splink blocking and comparison infrastructure
2. Frequency / uniqueness-based PUBLIM score
3. Strong / Fair / Weak confidence labels
4. Audit-friendly output

Install
-------
pip install splink duckdb pandas numpy rapidfuzz

Notes
-----
Splink APIs have changed across versions. This file uses a defensive pattern:

- It provides a Splink settings generator.
- It tries the common DuckDBLinker API.
- If Splink prediction output format changes, you can still use the
  score_publim_candidates() function directly on any candidate-pair dataframe.

Expected Dataset A columns:
    id, first_name, last_name, dob, yob, gender, zip5, state, admin_id

Expected Dataset B columns:
    id, first_name, last_name, dob, yob, gender, zip5, state, admin_id

You can rename your source columns before calling the functions.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Splink settings
# ---------------------------------------------------------------------

def build_splink_settings(
    unique_id_col: str = "id",
    source_dataset_col: str = "source_dataset",
) -> dict:
    """
    Build Splink settings for linking Dataset A to Dataset B.

    This uses Splink for candidate generation and traditional comparison
    features. PUBLIM scoring is added after Splink generates candidate pairs.

    You can adjust blocking rules depending on your file quality.
    """
    return {
        "link_type": "link_only",
        "unique_id_column_name": unique_id_col,
        "source_dataset_column_name": source_dataset_col,

        # Candidate generation rules.
        # Keep these relatively broad so PUBLIM can score multiple candidates.
        "blocking_rules_to_generate_predictions": [
            "l.admin_id = r.admin_id",
            "l.dob = r.dob and l.zip5 = r.zip5",
            "l.last_name = r.last_name and l.yob = r.yob",
            "l.first_name = r.first_name and l.last_name = r.last_name",
            "l.zip5 = r.zip5 and l.yob = r.yob",
        ],

        # Splink comparisons are still useful for diagnostics and dashboards.
        # Exact comparison settings are the most API-stable across Splink versions.
        "comparisons": [
            {"output_column_name": "admin_id", "comparison_levels": [
                {"sql_condition": "l.admin_id = r.admin_id", "label_for_charts": "Exact match"},
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ]},
            {"output_column_name": "first_name", "comparison_levels": [
                {"sql_condition": "l.first_name = r.first_name", "label_for_charts": "Exact match"},
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ]},
            {"output_column_name": "last_name", "comparison_levels": [
                {"sql_condition": "l.last_name = r.last_name", "label_for_charts": "Exact match"},
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ]},
            {"output_column_name": "dob", "comparison_levels": [
                {"sql_condition": "l.dob = r.dob", "label_for_charts": "Exact match"},
                {"sql_condition": "l.yob = r.yob", "label_for_charts": "YOB match"},
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ]},
            {"output_column_name": "zip5", "comparison_levels": [
                {"sql_condition": "l.zip5 = r.zip5", "label_for_charts": "Exact match"},
                {"sql_condition": "l.state = r.state", "label_for_charts": "State match"},
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ]},
            {"output_column_name": "gender", "comparison_levels": [
                {"sql_condition": "l.gender = r.gender", "label_for_charts": "Exact match"},
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ]},
        ],
    }


def run_splink_candidate_generation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    settings: Optional[dict] = None,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Run Splink to produce candidate pairs.

    Returns a pandas dataframe with left/right columns.

    If this fails due to Splink API version changes, use
    make_rule_based_candidates() as a fallback and still run PUBLIM scoring.
    """
    if settings is None:
        settings = build_splink_settings(unique_id_col=id_col)

    df_a = df_a.copy()
    df_b = df_b.copy()

    df_a["source_dataset"] = "A"
    df_b["source_dataset"] = "B"

    df_all = pd.concat([df_a, df_b], ignore_index=True)

    try:
        # Splink 3.x common import path
        from splink.duckdb.duckdb_linker import DuckDBLinker

        linker = DuckDBLinker(df_all, settings)

        # Optional training can be added here if needed.
        # For PUBLIM scoring, Splink primarily supplies candidate pairs.
        pred = linker.predict()
        out = pred.as_pandas_dataframe()

        return out

    except Exception as exc:
        raise RuntimeError(
            "Splink candidate generation failed. This is often due to Splink API "
            "version changes. You can still use make_rule_based_candidates() "
            "and then score_publim_candidates(). Original error: "
            f"{exc}"
        )


# ---------------------------------------------------------------------
# 2. Fallback candidate generation without Splink
# ---------------------------------------------------------------------

def make_rule_based_candidates(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Fallback candidate generator that mimics Splink-style blocking.

    Useful for testing PUBLIM scoring without depending on Splink.
    """
    blocks = [
        ["admin_id"],
        ["dob", "zip5"],
        ["last_name", "yob"],
        ["first_name", "last_name"],
        ["zip5", "yob"],
    ]

    pairs = []

    a = df_a.copy()
    b = df_b.copy()

    for block in blocks:
        available = [c for c in block if c in a.columns and c in b.columns]
        if len(available) != len(block):
            continue

        merged = a.merge(
            b,
            on=available,
            how="inner",
            suffixes=("_l", "_r"),
        )

        if merged.empty:
            continue

        for _, row in merged.iterrows():
            record = {}

            record["id_l"] = row.get(f"{id_col}_l", row.get(id_col))
            record["id_r"] = row.get(f"{id_col}_r", row.get(id_col))

            for col in set(a.columns).union(set(b.columns)):
                if col == id_col:
                    continue

                record[f"{col}_l"] = row.get(f"{col}_l", row.get(col))
                record[f"{col}_r"] = row.get(f"{col}_r", row.get(col))

            record["blocking_rule"] = " AND ".join(available)
            pairs.append(record)

    if not pairs:
        return pd.DataFrame(columns=["id_l", "id_r"])

    return pd.DataFrame(pairs).drop_duplicates(subset=["id_l", "id_r"])


# ---------------------------------------------------------------------
# 3. PUBLIM frequency tables
# ---------------------------------------------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    return s.fillna("__MISSING__").astype(str).str.strip().str.lower()


def build_frequency_tables(
    df_b: pd.DataFrame,
    fields: Iterable[str],
) -> Dict[str, pd.Series]:
    """
    Compute p(value) = share of Dataset B with that value.

    Dataset B is the reference population, consistent with the PUBLIM idea:
    uniqueness is measured against the larger/linking dataset.
    """
    n = len(df_b)
    if n == 0:
        raise ValueError("df_b is empty.")

    freq = {}
    for field in fields:
        if field in df_b.columns:
            freq[field] = normalize_series(df_b[field]).value_counts(dropna=False) / n

    return freq


def lookup_prob(freq: Dict[str, pd.Series], field: str, value, default: float = 1.0) -> float:
    if field not in freq:
        return default

    key = "__MISSING__" if pd.isna(value) else str(value).strip().lower()
    return float(freq[field].get(key, default))


# ---------------------------------------------------------------------
# 4. PUBLIM category probability logic
# ---------------------------------------------------------------------

CATEGORY_FIELDS = {
    "admin": ["admin_id"],
    "name": ["first_name", "last_name"],
    "age": ["dob", "yob"],
    "residence": ["zip5", "state"],
    "gender": ["gender"],
}


def get_pair_value(row: pd.Series, field: str, side: str):
    """
    Works with Splink-style columns:
        first_name_l, first_name_r
    """
    return row.get(f"{field}_{side}")


def exact_pair(row: pd.Series, field: str) -> bool:
    l = get_pair_value(row, field, "l")
    r = get_pair_value(row, field, "r")
    if pd.isna(l) or pd.isna(r):
        return False
    return str(l).strip().lower() == str(r).strip().lower()


def category_match_quality(row: pd.Series, category: str) -> Tuple[str, int]:
    """
    Approximate match-quality band labeling.

    Returns:
        (match_label, band)

    Lower band = stronger match.
    """
    if category == "admin":
        if exact_pair(row, "admin_id"):
            return "admin_exact", 1
        return "admin_missing_or_mismatch", 4

    if category == "name":
        fn = exact_pair(row, "first_name")
        ln = exact_pair(row, "last_name")
        if fn and ln:
            return "full_name_exact", 1
        if ln:
            return "last_name_only", 3
        return "name_missing_or_mismatch", 5

    if category == "age":
        if exact_pair(row, "dob"):
            return "dob_exact", 1
        if exact_pair(row, "yob"):
            return "yob_exact", 3
        return "age_missing_or_mismatch", 7

    if category == "residence":
        if exact_pair(row, "zip5"):
            return "zip5_exact", 1
        if exact_pair(row, "state"):
            return "state_exact", 3
        return "residence_missing_or_mismatch", 4

    if category == "gender":
        if exact_pair(row, "gender"):
            return "gender_exact", 1
        return "gender_missing_or_mismatch", 2

    return "unknown", 99


def category_probability(
    row: pd.Series,
    category: str,
    freq: Dict[str, pd.Series],
) -> float:
    """
    Compute p_k for one category.

    Conservative approach:
    - Only fields that match contribute uniqueness.
    - Within a category, use the most informative available combined logic.
    - Missing/mismatch returns 1.0, meaning no uniqueness contribution.

    You can make this more aggressive by multiplying within category, but that
    risks double-counting correlated variables.
    """
    label, band = category_match_quality(row, category)

    if "missing_or_mismatch" in label:
        return 1.0

    if category == "admin" and exact_pair(row, "admin_id"):
        return lookup_prob(freq, "admin_id", get_pair_value(row, "admin_id", "r"))

    if category == "name":
        if exact_pair(row, "first_name") and exact_pair(row, "last_name"):
            # Full name frequency approximation:
            # p(first,last) is estimated from product of marginals for simplicity.
            # Production version should precompute full-name frequency.
            p_fn = lookup_prob(freq, "first_name", get_pair_value(row, "first_name", "r"))
            p_ln = lookup_prob(freq, "last_name", get_pair_value(row, "last_name", "r"))
            return min(1.0, p_fn * p_ln)

        if exact_pair(row, "last_name"):
            return lookup_prob(freq, "last_name", get_pair_value(row, "last_name", "r"))

    if category == "age":
        if exact_pair(row, "dob"):
            return lookup_prob(freq, "dob", get_pair_value(row, "dob", "r"))
        if exact_pair(row, "yob"):
            return lookup_prob(freq, "yob", get_pair_value(row, "yob", "r"))

    if category == "residence":
        if exact_pair(row, "zip5"):
            return lookup_prob(freq, "zip5", get_pair_value(row, "zip5", "r"))
        if exact_pair(row, "state"):
            return lookup_prob(freq, "state", get_pair_value(row, "state", "r"))

    if category == "gender" and exact_pair(row, "gender"):
        return lookup_prob(freq, "gender", get_pair_value(row, "gender", "r"))

    return 1.0


# ---------------------------------------------------------------------
# 5. PUBLIM score and confidence
# ---------------------------------------------------------------------

def publim_score_row(
    row: pd.Series,
    freq: Dict[str, pd.Series],
    categories: Optional[List[str]] = None,
    epsilon: float = 1e-15,
) -> pd.Series:
    """
    Compute PUBLIM-style score for a candidate pair.

    P = product of p_k across categories
    odds = (1 - P) / P
    score = log10(odds)
    """
    if categories is None:
        categories = list(CATEGORY_FIELDS.keys())

    p_values = {}
    bands = {}

    for category in categories:
        label, band = category_match_quality(row, category)
        p = category_probability(row, category, freq)
        p_values[f"p_{category}"] = max(min(p, 1.0), epsilon)
        bands[f"band_{category}"] = band
        bands[f"match_{category}"] = label

    combined_p = float(np.prod(list(p_values.values())))
    odds = (1.0 - combined_p) / max(combined_p, epsilon)
    score = math.log10(max(odds, epsilon))

    return pd.Series({
        **p_values,
        **bands,
        "publim_combined_probability": combined_p,
        "publim_score": score,
    })


def missing_data_pattern(row: pd.Series, fields: Iterable[str]) -> str:
    """
    Produce a compact missing-data pattern for threshold calibration.
    Example: admin_id:missing|dob:present|zip5:present
    """
    parts = []
    for field in fields:
        l_missing = pd.isna(get_pair_value(row, field, "l"))
        r_missing = pd.isna(get_pair_value(row, field, "r"))
        status = "missing" if l_missing or r_missing else "present"
        parts.append(f"{field}:{status}")
    return "|".join(parts)


def assign_confidence(
    score: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Placeholder thresholds.

    Replace these with thresholds calibrated from false-match simulation:
    - Strong: <1% estimated false-match rate
    - Fair:   1–5%
    - Weak:   5–10%
    """
    if thresholds is None:
        thresholds = {
            "strong": 12.0,
            "fair": 10.7,
            "weak": 9.0,
        }

    if score >= thresholds["strong"]:
        return "Strong"
    if score >= thresholds["fair"]:
        return "Fair"
    if score >= thresholds["weak"]:
        return "Weak"
    return "No Match"


def score_publim_candidates(
    candidate_pairs: pd.DataFrame,
    df_b: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    id_l_col: str = "id_l",
    id_r_col: str = "id_r",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score Splink candidate pairs with PUBLIM.

    Returns:
        all_scored_pairs:
            every candidate pair with PUBLIM score

        best_links:
            one best candidate per Dataset A ID, with score gap and confidence
    """
    fields = sorted({f for fields in CATEGORY_FIELDS.values() for f in fields})
    freq = build_frequency_tables(df_b, fields)

    scored_extra = candidate_pairs.apply(lambda r: publim_score_row(r, freq), axis=1)
    scored = pd.concat([candidate_pairs.reset_index(drop=True), scored_extra], axis=1)

    scored["missing_pattern"] = scored.apply(
        lambda r: missing_data_pattern(r, ["admin_id", "dob", "zip5"]),
        axis=1,
    )

    scored = scored.sort_values([id_l_col, "publim_score"], ascending=[True, False])

    # Rank candidates within each Dataset A record.
    scored["candidate_rank"] = scored.groupby(id_l_col).cumcount() + 1

    # Compute score gap between best and second best candidate.
    second_scores = (
        scored[scored["candidate_rank"] == 2]
        [[id_l_col, "publim_score"]]
        .rename(columns={"publim_score": "second_best_score"})
    )

    best = scored[scored["candidate_rank"] == 1].copy()
    best = best.merge(second_scores, on=id_l_col, how="left")
    best["score_gap"] = best["publim_score"] - best["second_best_score"].fillna(-999)

    best["confidence"] = best["publim_score"].apply(lambda x: assign_confidence(x, thresholds))

    # Conservative review flag:
    # Even a high score should be reviewed when the top two candidates are close.
    best["review_flag"] = np.where(
        (best["confidence"] != "No Match") & (best["score_gap"] < 1.0),
        True,
        False,
    )

    best["matched_id_b"] = np.where(
        best["confidence"] == "No Match",
        None,
        best[id_r_col],
    )

    return scored, best


# ---------------------------------------------------------------------
# 6. End-to-end helper
# ---------------------------------------------------------------------

def publim_on_splink(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    use_splink: bool = True,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end workflow.

    If use_splink=True, attempt Splink candidate generation.
    If Splink fails or use_splink=False, use fallback blocking rules.
    """
    if use_splink:
        try:
            candidates = run_splink_candidate_generation(df_a, df_b)
        except Exception as exc:
            print("WARNING: Falling back to rule-based candidates.")
            print(exc)
            candidates = make_rule_based_candidates(df_a, df_b)
    else:
        candidates = make_rule_based_candidates(df_a, df_b)

    if candidates.empty:
        raise ValueError("No candidate pairs generated. Add broader blocking rules.")

    scored_pairs, best_links = score_publim_candidates(
        candidates,
        df_b=df_b,
        thresholds=thresholds,
    )

    return scored_pairs, best_links


# ---------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df_a = pd.DataFrame({
        "id": ["A1", "A2", "A3", "A4"],
        "first_name": ["James", "Robert", "Dion", "Maya"],
        "last_name": ["Smith", "Martines", "Wooden", "Delgado"],
        "dob": ["1943-01-01", "1951-01-01", "1947-01-01", "1930-01-01"],
        "yob": [1943, 1951, 1947, 1930],
        "gender": ["M", "M", "M", "F"],
        "zip5": ["59921", "77449", "77449", "11368"],
        "state": ["MT", "TX", "TX", "NY"],
        "admin_id": [None, None, None, None],
    })

    df_b = pd.DataFrame({
        "id": ["B1", "B2", "B3", "B4", "B5"],
        "first_name": ["Jim", "James", "Robert", "Dion", "Mary"],
        "last_name": ["Smyth", "Smith", "Martines", "Wooden", "Delapaz"],
        "dob": ["1943-01-01", "1943-01-01", "1952-01-01", "1947-01-01", "1930-01-01"],
        "yob": [1943, 1943, 1952, 1947, 1930],
        "gender": ["M", "M", "M", "M", "F"],
        "zip5": ["59921", "77449", "77449", "11368", "77449"],
        "state": ["MT", "TX", "TX", "NY", "TX"],
        "admin_id": [None, None, None, None, None],
    })

    # Use fallback mode for a guaranteed demo run.
    scored, best = publim_on_splink(df_a, df_b, use_splink=False)

    print("\nBest links:")
    print(best[[
        "id_l",
        "matched_id_b",
        "publim_score",
        "publim_combined_probability",
        "confidence",
        "score_gap",
        "review_flag",
        "missing_pattern",
    ]])

    print("\nAll scored candidate pairs:")
    print(scored[[
        "id_l",
        "id_r",
        "publim_score",
        "p_name",
        "p_age",
        "p_residence",
        "p_gender",
        "candidate_rank",
    ]])
