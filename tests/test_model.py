"""Tests for the model pipeline, MAE helpers and training logic.

They build small data inline (no CSVs, no MLflow server needed), so they run in CI.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from solar_ml.config import load_params
from solar_ml.evaluate import evaluate, mae, mae_by_group
from solar_ml.model import build_pipeline, feature_columns, make_regressor
from solar_ml.train import compare_candidates, fit_final, update_findings_tracking

PARAMS = load_params()


def make_clean(n=120, seed=0):
    """Clean synthetic rows where energy depends linearly on the features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "panel_area_m2": rng.uniform(8, 60, n).round(1),
        "sunshine_hours": rng.uniform(2, 11, n).round(1),
        "temp_c": rng.uniform(12, 46, n).round(1),
        "dust_days": rng.integers(0, 31, n),
        "panel_age_yr": rng.uniform(0, 15, n).round(1),
        "cloud_cover_pct": rng.integers(0, 91, n),
        "roof_type": rng.choice(["flat", "sloped", "tin"], n),
        "area": rng.choice(["Old City", "New Colony", "Industrial"], n),
    })
    df["energy_kwh"] = (40 + 0.1 * df.panel_area_m2 - 1.2 * df.panel_age_yr
                        - 0.15 * df.cloud_cover_pct - 0.2 * df.dust_days
                        + rng.normal(0, 0.5, n)).round(2)
    return df


def test_mae_is_average_absolute_error():
    assert mae([10, 20, 30], [12, 18, 30]) == pytest.approx(4 / 3)
    assert mae([5.0], [5.0]) == 0.0


def test_feature_columns_match_api_fields():
    assert feature_columns(PARAMS) == [
        "panel_area_m2", "sunshine_hours", "temp_c", "dust_days",
        "panel_age_yr", "cloud_cover_pct", "roof_type", "area"]


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError):
        make_regressor("xgboost_magic")


@pytest.mark.parametrize("kind,hp", [
    ("ridge", {"alpha": 1.0}),
    ("random_forest", {"n_estimators": 20}),
    ("gradient_boosting", {"n_estimators": 30}),
])
def test_pipeline_fits_and_predicts_on_raw_fields(kind, hp):
    df = make_clean()
    pipe = build_pipeline(kind, hp, PARAMS).fit(df[feature_columns(PARAMS)], df["energy_kwh"])
    preds = pipe.predict(df[feature_columns(PARAMS)].head(7))
    assert preds.shape == (7,) and np.isfinite(preds).all()


def test_pipeline_handles_unseen_category():
    df = make_clean()
    pipe = build_pipeline("ridge", {"alpha": 1.0}, PARAMS)
    pipe.fit(df[feature_columns(PARAMS)], df["energy_kwh"])
    row = df[feature_columns(PARAMS)].head(1).assign(roof_type="thatch")
    assert np.isfinite(pipe.predict(row)).all()


def test_compare_candidates_picks_lowest_cv_mae():
    df = make_clean(200)
    results, best = compare_candidates(df, PARAMS)
    assert len(results) == len(PARAMS["model"]["candidates"]) >= 3
    assert best["cv_mae"] == min(r["cv_mae"] for r in results)
    # the data is linear, so a ridge setting should win with a small error
    assert best["type"] == "ridge" and best["mae_val"] < 1.0


def test_final_model_round_trips_through_joblib(tmp_path):
    df = make_clean()
    _, best = compare_candidates(df, PARAMS)
    model = fit_final(best, df, PARAMS)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert evaluate(loaded, df, PARAMS) == pytest.approx(evaluate(model, df, PARAMS))
    groups = mae_by_group(loaded, df, PARAMS)
    assert set(groups) == {"Old City", "New Colony", "Industrial"}


def test_update_findings_tracking_keeps_comments(tmp_path):
    f = tmp_path / "findings.yaml"
    f.write_text(
        "data_issues:\n  duplicate_rows:\n    count: 3\n"
        "tracking:\n"
        "  experiment_name: null\n"
        "  best_run_id: null        # the MLflow run id of your v1 model\n"
        "  registered_model_name: null\n"
        "  champion_version: null\n"
        "monitoring:\n  decision: null\n", encoding="utf-8")
    update_findings_tracking(f, {"experiment_name": "exp", "best_run_id": "abc123",
                                 "registered_model_name": "m", "champion_version": 1})
    text = f.read_text(encoding="utf-8")
    assert 'best_run_id: "abc123"        # the MLflow run id' in text
    assert "champion_version: 1\n" in text
    assert "decision: null" in text  # other sections untouched
