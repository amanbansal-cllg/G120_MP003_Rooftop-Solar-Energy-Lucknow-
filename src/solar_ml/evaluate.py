"""MAE evaluation helpers, shared by training (Stage 2) and monitoring (Stage 4).

Run:  python -m solar_ml.evaluate        (from the repo root, with PYTHONPATH=src)
      prints the saved model's MAE on cleaned train.csv, overall and per area.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from solar_ml.config import load_params, resolve
from solar_ml.model import split_xy


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: the average size of a mistake, in kWh."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate(model, df: pd.DataFrame, params: dict) -> float:
    X, y = split_xy(df, params)
    return mae(y, model.predict(X))


def mae_by_group(model, df: pd.DataFrame, params: dict,
                 group_col: str | None = None) -> dict[str, float]:
    """MAE separately for each value of the group column (area by default)."""
    group_col = group_col or params["columns"]["group"]
    X, y = split_xy(df, params)
    err = pd.Series(np.abs(y.to_numpy() - model.predict(X)), index=df.index)
    return {str(g): round(float(e.mean()), 4) for g, e in err.groupby(df[group_col])}


def load_model(path: str | Path | None = None):
    """Load the service model (models/model.joblib by default)."""
    params = load_params()
    return joblib.load(path or resolve(params["model"]["output_path"]))


def load_champion(params: dict | None = None):
    """Load the @champion model straight from the MLflow registry."""
    import mlflow  # imported here so tests and the API do not need MLflow

    params = params or load_params()
    m = params["mlflow"]
    mlflow.set_tracking_uri(m["tracking_uri"])
    return mlflow.sklearn.load_model(f"models:/{m['registered_model_name']}@{m['champion_alias']}")


def main() -> int:
    params = load_params()
    df = pd.read_csv(resolve(params["data"]["cleaned_train"]))
    model = load_model()
    print(f"MAE on cleaned train.csv: {evaluate(model, df, params):.3f} kWh")
    print("MAE by area:", mae_by_group(model, df, params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
