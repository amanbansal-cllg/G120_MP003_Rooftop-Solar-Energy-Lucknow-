"""Load project settings from params.yaml and the raw data files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Turn a path from params.yaml into an absolute path in the repo."""
    return ROOT / rel_path


def load_csv(name: str, params: dict | None = None) -> pd.DataFrame:
    """Load one of the data files named in params['data'] (train, later, holdout)."""
    params = params or load_params()
    return pd.read_csv(resolve(params["data"][name]))


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, logged to MLflow as data_sha256."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
