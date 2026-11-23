"""Build the scikit-learn pipeline used for every model in the project.

The pipeline takes the eight raw request fields (the same ones the API gets)
and does its own preprocessing, so the saved model can be called directly on
a DataFrame built from a POST /predict body:

    numeric columns     -> StandardScaler
    categorical columns -> OneHotEncoder(handle_unknown="ignore")
    then a regressor chosen by `type` in params.yaml
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REGRESSORS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def feature_columns(params: dict) -> list[str]:
    """The model's inputs, in order: numeric then categorical (no id, date or target)."""
    cols = params["columns"]
    return list(cols["numeric"]) + list(cols["categorical"])


def make_regressor(kind: str, hyperparams: dict | None = None, random_state: int = 42):
    if kind not in REGRESSORS:
        raise ValueError(f"unknown model type {kind!r}; choose from {sorted(REGRESSORS)}")
    hp = dict(hyperparams or {})
    if kind != "ridge":
        hp.setdefault("random_state", random_state)
    return REGRESSORS[kind](**hp)


def build_pipeline(kind: str, hyperparams: dict | None, params: dict,
                   random_state: int = 42) -> Pipeline:
    cols = params["columns"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(cols["numeric"])),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(cols["categorical"])),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("model", make_regressor(kind, hyperparams, random_state)),
    ])


def split_xy(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns(params)], df[params["columns"]["target"]]
