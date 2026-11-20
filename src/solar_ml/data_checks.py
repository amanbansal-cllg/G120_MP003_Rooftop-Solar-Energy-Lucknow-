"""Detect the seven kinds of data problem described in STUDENT_HANDBOOK.md section 7.

Every function takes a DataFrame and returns a dict in the same shape as the
`data_issues` block of reports/findings.yaml. Counts are on train.csv as given.

Run:  python -m solar_ml.data_checks        (from the repo root, with PYTHONPATH=src)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from solar_ml.config import load_csv, load_params, resolve


def _absent(*keys: str) -> dict:
    out = {"present": False}
    out.update({k: None for k in keys})
    return out


# 1. leaking column -----------------------------------------------------------
def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str,
                         id_col: str, corr_threshold: float = 0.95) -> dict:
    """A feature that is missing from the holdout file, or that almost equals the target."""
    candidates = [c for c in train.columns if c not in (target, id_col)]
    missing_in_holdout = [c for c in candidates if c not in holdout.columns]
    if missing_in_holdout:
        return {"present": True, "column": missing_in_holdout[0]}
    numeric = train[candidates].select_dtypes("number")
    corr = numeric.corrwith(train[target]).abs()
    strong = corr[corr >= corr_threshold]
    if not strong.empty:
        return {"present": True, "column": strong.idxmax()}
    return _absent("column")


# 2. unit mix-up --------------------------------------------------------------
def check_unit_mixup(df: pd.DataFrame, group_col: str, numeric_cols: list[str],
                     factors: list[float], ratio_min: float = 5,
                     sentinels: list[float] = ()) -> dict:
    """One group's values of one column are a fixed unit factor bigger than the rest."""
    best = None
    for col in numeric_cols:
        values = df[col].where(~df[col].isin(list(sentinels)))  # ignore 999-style sentinels
        for g in df[group_col].dropna().unique():
            inside = values[df[group_col] == g].median()
            outside = values[df[group_col] != g].median()
            if not inside or not outside or np.isnan(inside) or np.isnan(outside):
                continue
            ratio = max(inside / outside, outside / inside)
            if ratio < ratio_min:
                continue
            factor = min(factors, key=lambda f: abs(np.log(ratio / f)))
            if best is None or ratio > best[0]:
                best = (ratio, col, g, factor)
    if best is None:
        return _absent("column", "group_value", "factor")
    _, col, g, factor = best
    return {"present": True, "column": col, "group_value": str(g), "factor": int(factor)}


# 3. sentinel value -----------------------------------------------------------
def check_sentinel_value(df: pd.DataFrame, numeric_cols: list[str],
                         candidates: list[float], min_count: int = 5) -> dict:
    """An impossible number such as 999 typed many times instead of a blank."""
    best = None
    for col in numeric_cols:
        counts = df[col].value_counts()
        for v in candidates:
            n = int(counts.get(v, 0))
            if n >= min_count and (best is None or n > best[2]):
                best = (col, v, n)
    if best is None:
        return _absent("column", "value", "count")
    col, v, n = best
    return {"present": True, "column": col, "value": int(v), "count": n}


# 4. duplicate rows -----------------------------------------------------------
def check_duplicate_rows(df: pd.DataFrame) -> dict:
    n = int(df.duplicated().sum())
    return {"present": n > 0, "count": n if n else None}


# 5. stuck sensor -------------------------------------------------------------
def longest_run(df: pd.DataFrame, col: str, date_col: str) -> tuple[int, str, str, float]:
    """Longest run of identical consecutive values after sorting by date.

    Returns (number of distinct days in the run, first date, last date, value).
    """
    d = df.drop_duplicates().sort_values([date_col], kind="stable")
    run_id = (d[col] != d[col].shift()).cumsum()
    runs = d.groupby(run_id).agg(value=(col, "first"),
                                 start=(date_col, "first"),
                                 end=(date_col, "last"),
                                 days=(date_col, "nunique"))
    top = runs.sort_values("days").iloc[-1]
    return int(top["days"]), str(top["start"]), str(top["end"]), float(top["value"])


def check_stuck_sensor(df: pd.DataFrame, numeric_cols: list[str], date_col: str,
                       min_run_days: int = 7) -> dict:
    best = None
    for col in numeric_cols:
        days, start, end, value = longest_run(df, col, date_col)
        if days >= min_run_days and (best is None or days > best[0]):
            best = (days, col, start, end)
    if best is None:
        return _absent("column", "start_date", "end_date")
    _, col, start, end = best
    return {"present": True, "column": col, "start_date": start, "end_date": end}


# 6. target typos -------------------------------------------------------------
def find_target_typos(target: pd.Series, factor: float = 10,
                      gap_ratio: float = 2.0) -> pd.Series:
    """Boolean mask of targets that sit above a big empty gap in the sorted values.

    The gap is the largest jump between neighbouring sorted values; it must be at
    least `gap_ratio` times (value above / value below), and the values above it
    must be close to `factor` times the normal range.
    """
    s = target.dropna().sort_values()
    if len(s) < 3:
        return pd.Series(False, index=target.index)
    ratios = (s.shift(-1) / s).iloc[:-1]
    i = int(np.argmax(ratios.values))
    if ratios.iloc[i] < gap_ratio:
        return pd.Series(False, index=target.index)
    cutoff = s.iloc[i]
    high = target > cutoff
    # Sanity check: the high group should look like normal values x factor.
    if high.sum() and target[high].median() / target[~high].median() < factor / 3:
        return pd.Series(False, index=target.index)
    return high


def check_target_typos(df: pd.DataFrame, target: str, factor: float = 10,
                       gap_ratio: float = 2.0) -> dict:
    n = int(find_target_typos(df[target], factor, gap_ratio).sum())
    return {"present": n > 0, "count": n if n else None}


# 7. spelling variants --------------------------------------------------------
def normalise_label(value: object) -> object:
    """'  Sloped', 'SLOPED', 'slo.' -> 'slo' (first three letters, lower case)."""
    if not isinstance(value, str):
        return value
    return value.strip().lower().rstrip(".")[:3]


def check_spelling_variants(df: pd.DataFrame, text_cols: list[str],
                            reference: pd.DataFrame | None = None) -> dict:
    """A text column with more raw spellings than real categories."""
    best = None
    for col in text_cols:
        raw = df[col].dropna().nunique()
        if reference is not None and col in reference:
            true = reference[col].dropna().nunique()
        else:
            true = df[col].dropna().map(normalise_label).nunique()
        if raw > true and (best is None or raw - true > best[2]):
            best = (col, true, raw - true)
    if best is None:
        return _absent("column", "true_levels")
    col, true, _ = best
    return {"present": True, "column": col, "true_levels": int(true)}


# all seven -------------------------------------------------------------------
def run_all_checks(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame,
                   params: dict) -> dict:
    cols, cfg = params["columns"], params["data_checks"]
    return {
        "leaking_column": check_leaking_column(train, holdout, cols["target"], cols["id"],
                                               cfg["leak_corr_threshold"]),
        "unit_mixup": check_unit_mixup(train, cols["group"], cols["numeric"],
                                       cfg["unit_factors"], cfg["unit_ratio_min"],
                                       cfg["sentinel_candidates"]),
        "sentinel_value": check_sentinel_value(train, cols["numeric"],
                                               cfg["sentinel_candidates"],
                                               cfg["sentinel_min_count"]),
        "duplicate_rows": check_duplicate_rows(train),
        "stuck_sensor": check_stuck_sensor(train, cols["numeric"], cols["date"],
                                           cfg["stuck_min_run_days"]),
        "target_typos": check_target_typos(train, cols["target"],
                                           cfg["target_typo_factor"],
                                           cfg["target_typo_gap_ratio"]),
        "spelling_variants": check_spelling_variants(train, cols["categorical"], later),
    }


def main() -> int:
    params = load_params()
    train, later, holdout = (load_csv(n, params) for n in ("train", "later", "holdout"))
    results = run_all_checks(train, later, holdout, params)
    out = resolve(params["reports"]["data_checks"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    json.dump(results, sys.stdout, indent=2)
    print(f"\nSaved to {out.relative_to(resolve('.'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
