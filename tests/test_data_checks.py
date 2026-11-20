"""Tests for the seven data checks. They build small data inline (no CSVs needed)."""

import pandas as pd

from solar_ml import data_checks as dc
from solar_ml.cleaning import clean_train
from solar_ml.config import load_params


def make_df(n=40):
    dates = pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": dates,
        "panel_area_m2": [20.0 + (i % 7) for i in range(n)],
        "sunshine_hours": [4.0 + (i % 5) for i in range(n)],
        "temp_c": [25.0 + (i % 9) for i in range(n)],
        "dust_days": [i % 11 for i in range(n)],
        "panel_age_yr": [1.0 + (i % 6) for i in range(n)],
        "cloud_cover_pct": [10 * (i % 8) for i in range(n)],
        "roof_type": ["flat", "sloped", "tin", "flat"] * (n // 4),
        "area": ["Old City", "New Colony", "Industrial", "Old City"] * (n // 4),
        "energy_kwh": [20.0 + (i % 13) for i in range(n)],
    })


def test_clean_data_has_no_problems():
    df = make_df()
    holdout = df.drop(columns="energy_kwh")
    params = load_params()
    issues = dc.run_all_checks(df, df, holdout, params)
    assert all(v["present"] is False for v in issues.values()), issues


def test_duplicates_and_sentinel():
    df = make_df()
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    df.loc[[5, 9, 12, 15, 20, 25], "sunshine_hours"] = 999
    assert dc.check_duplicate_rows(df) == {"present": True, "count": 3}
    s = dc.check_sentinel_value(df, ["sunshine_hours", "temp_c"], [999, -1])
    assert s["present"] and s["column"] == "sunshine_hours" and s["value"] == 999


def test_unit_mixup_factor():
    df = make_df()
    rows = df["area"] == "New Colony"
    df.loc[rows, "panel_area_m2"] *= 10000
    u = dc.check_unit_mixup(df, "area", ["panel_area_m2", "temp_c"], [10, 100, 1000, 10000])
    assert u == {"present": True, "column": "panel_area_m2",
                 "group_value": "New Colony", "factor": 10000}


def test_target_typos_and_spelling():
    df = make_df()
    df.loc[[3, 7], "energy_kwh"] *= 10
    df.loc[[1, 2], "roof_type"] = [" Sloped", "TIN"]
    assert dc.check_target_typos(df, "energy_kwh")["count"] == 2
    sv = dc.check_spelling_variants(df, ["roof_type", "area"])
    assert sv == {"present": True, "column": "roof_type", "true_levels": 3}


def test_stuck_sensor_and_leak():
    df = make_df()
    df.loc[10:20, "temp_c"] = 31.5
    st = dc.check_stuck_sensor(df, ["temp_c", "dust_days"], "date", 7)
    assert st == {"present": True, "column": "temp_c",
                  "start_date": "2026-01-11", "end_date": "2026-01-21"}
    df["metered_kwh"] = df["energy_kwh"] * 1.01
    leak = dc.check_leaking_column(df, df.drop(columns=["energy_kwh", "metered_kwh"]),
                                   "energy_kwh", "record_id")
    assert leak == {"present": True, "column": "metered_kwh"}


def test_clean_train_fixes_everything():
    df = make_df()
    clean_ref = df.copy()
    dirty = pd.concat([df, df.iloc[:2]], ignore_index=True)
    dirty.loc[dirty["area"] == "New Colony", "panel_area_m2"] *= 10000
    dirty.loc[[4, 8, 16, 24, 28], "sunshine_hours"] = 999
    dirty.loc[[3], "energy_kwh"] *= 10
    dirty.loc[[5], "roof_type"] = "SLO."
    out = clean_train(dirty, clean_ref, clean_ref.drop(columns="energy_kwh"), load_params())
    assert len(out) == len(df)
    assert out["panel_area_m2"].max() < 100
    assert (out["sunshine_hours"] != 999).all()
    assert out["energy_kwh"].max() < 100
    assert set(out["roof_type"]) == {"flat", "sloped", "tin"}
