"""Clean train.csv using the problems found by data_checks.

Cleaned-data contract (hand-off to the model lead):
  * one row per real record (extra duplicate copies removed)
  * roof_type spelled exactly as in later.csv: flat, sloped, tin
  * panel_area_m2 in square metres for every area
  * sentinel values replaced by the median of the real readings
  * rows with a target typed 10x too big are divided back by 10
  * no leaking column is used as a feature
  * features = params.columns.numeric + params.columns.categorical; target = energy_kwh

Run:  python -m solar_ml.cleaning        (from the repo root, with PYTHONPATH=src)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solar_ml.config import load_csv, load_params, resolve
from solar_ml.data_checks import find_target_typos, normalise_label, run_all_checks


def fix_spelling(df: pd.DataFrame, col: str, clean_levels: list[str]) -> pd.DataFrame:
    lookup = {normalise_label(level): level for level in clean_levels}
    out = df.copy()
    out[col] = out[col].map(lambda v: lookup.get(normalise_label(v), v))
    return out


def clean_train(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame,
                params: dict) -> pd.DataFrame:
    cols, cfg = params["columns"], params["data_checks"]
    issues = run_all_checks(train, later, holdout, params)
    df = train.copy()

    if issues["duplicate_rows"]["present"]:
        df = df.drop_duplicates().reset_index(drop=True)

    if issues["spelling_variants"]["present"]:
        col = issues["spelling_variants"]["column"]
        df = fix_spelling(df, col, sorted(later[col].dropna().unique()))

    if issues["unit_mixup"]["present"]:
        u = issues["unit_mixup"]
        rows = df[cols["group"]] == u["group_value"]
        df.loc[rows, u["column"]] = df.loc[rows, u["column"]] / u["factor"]

    if issues["sentinel_value"]["present"]:
        s = issues["sentinel_value"]
        df[s["column"]] = df[s["column"]].replace(s["value"], np.nan)
        df[s["column"]] = df[s["column"]].fillna(df[s["column"]].median())

    if issues["target_typos"]["present"]:
        typo = find_target_typos(df[cols["target"]], cfg["target_typo_factor"],
                                 cfg["target_typo_gap_ratio"])
        df.loc[typo, cols["target"]] = df.loc[typo, cols["target"]] / cfg["target_typo_factor"]

    if issues["leaking_column"]["present"]:
        df = df.drop(columns=[issues["leaking_column"]["column"]])

    return df


def main() -> int:
    params = load_params()
    train, later, holdout = (load_csv(n, params) for n in ("train", "later", "holdout"))
    cleaned = clean_train(train, later, holdout, params)
    out = resolve(params["data"]["cleaned_train"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out, index=False)
    print(f"{len(train)} rows in, {len(cleaned)} rows out -> {out.relative_to(resolve('.'))}")
    print(cleaned.describe().T[["min", "mean", "max"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
