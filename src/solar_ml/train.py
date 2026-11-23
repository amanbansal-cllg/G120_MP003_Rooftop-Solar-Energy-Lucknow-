"""Stage 2: train v1 with MLflow tracking and register it as @champion.

What it does, in order:
  1. Cleans train.csv with solar_ml.cleaning and saves data/processed/train_clean.csv.
  2. Holds back 20% of the cleaned rows. For every candidate setting in
     params.yaml it logs one MLflow run with the 5-fold CV MAE on the other
     80% (`cv_mae`) and the MAE on the held-back 20% (`mae_val`).
  3. Picks the setting with the lowest cv_mae, refits it on ALL cleaned rows,
     logs that as the run "v1_<name>", registers it (version 1) and gives it
     the alias @champion.
  4. Saves the same pipeline to models/model.joblib for the API, and writes
     the run id and registry details into reports/findings.yaml.

Every run logs the parameter data_sha256 (SHA-256 of the original train.csv).

Run once, from the repo root:   python -m solar_ml.train      (PYTHONPATH=src)
Use --force to register another version even if the model already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from solar_ml.cleaning import clean_train
from solar_ml.config import ROOT, file_sha256, load_csv, load_params, resolve
from solar_ml.evaluate import mae
from solar_ml.model import build_pipeline, split_xy


# --- pure scikit-learn part (tested without MLflow) ---------------------------
def score_candidate(cand: dict, train_part: pd.DataFrame, val_part: pd.DataFrame,
                    params: dict) -> dict:
    """Fit one candidate on train_part; return its CV MAE and validation MAE."""
    mcfg = params["model"]
    rs = mcfg.get("random_state", 42)
    X_tr, y_tr = split_xy(train_part, params)
    X_val, y_val = split_xy(val_part, params)

    pipe = build_pipeline(cand["type"], cand.get("params"), params, rs)
    folds = KFold(n_splits=mcfg.get("cv_folds", 5), shuffle=True, random_state=rs)
    cv = -cross_val_score(pipe, X_tr, y_tr, cv=folds, scoring="neg_mean_absolute_error")
    pipe.fit(X_tr, y_tr)
    return {
        "name": cand["name"],
        "type": cand["type"],
        "params": dict(cand.get("params") or {}),
        "cv_mae": float(np.mean(cv)),
        "cv_mae_std": float(np.std(cv)),
        "mae_val": mae(y_val, pipe.predict(X_val)),
    }


def compare_candidates(clean_df: pd.DataFrame, params: dict) -> tuple[list[dict], dict]:
    """Score every candidate on the same split. Returns (all results, best result)."""
    mcfg = params["model"]
    train_part, val_part = train_test_split(clean_df, test_size=mcfg.get("val_size", 0.2),
                                            random_state=mcfg.get("random_state", 42))
    results = [score_candidate(c, train_part, val_part, params) for c in mcfg["candidates"]]
    metric = mcfg.get("select_metric", "cv_mae")
    best = min(results, key=lambda r: r[metric])
    return results, best


def fit_final(best: dict, clean_df: pd.DataFrame, params: dict):
    """Refit the chosen setting on every cleaned row (this is v1)."""
    X, y = split_xy(clean_df, params)
    pipe = build_pipeline(best["type"], best["params"], params,
                          params["model"].get("random_state", 42))
    return pipe.fit(X, y)


def update_findings_tracking(path: Path, values: dict) -> None:
    """Fill the `tracking:` block of findings.yaml in place, keeping the comments."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    inside = False
    for i, line in enumerate(lines):
        if re.match(r"^tracking:\s*$", line):
            inside = True
            continue
        if inside and line.strip() and not line.startswith((" ", "\t")):
            break
        if inside:
            m = re.match(r"^(\s+)(\w+):(\s*)([^#\n]*?)(\s*#.*)?(\r?\n)?$", line)
            if m and m.group(2) in values:
                v = values[m.group(2)]
                text = str(v) if isinstance(v, int) else f'"{v}"'
                lines[i] = f"{m.group(1)}{m.group(2)}: {text}{m.group(5) or ''}{m.group(6) or ''}"
    path.write_text("".join(lines), encoding="utf-8")


# --- MLflow part --------------------------------------------------------------
def _log_model(mlflow, pipe, X: pd.DataFrame, registered_name: str):
    from mlflow.models import infer_signature

    sig = infer_signature(X.head(20), pipe.predict(X.head(20)))
    kwargs = dict(sk_model=pipe, name="model", signature=sig, input_example=X.head(5),
                  registered_model_name=registered_name)
    try:
        # MLflow 3.15+ refuses some sklearn types with its safer default format.
        return mlflow.sklearn.log_model(serialization_format="cloudpickle", **kwargs)
    except TypeError:
        return mlflow.sklearn.log_model(**kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="register a new version even if the model already has versions")
    args = ap.parse_args(argv)

    import mlflow
    from mlflow.tracking import MlflowClient

    os.chdir(ROOT)  # mlflow.db and the artifact folder live in the repo root
    params = load_params()
    mf, mcfg = params["mlflow"], params["model"]

    raw_path = resolve(params["data"]["train"])
    data_sha = file_sha256(raw_path)
    train, later, holdout = (load_csv(n, params) for n in ("train", "later", "holdout"))
    clean = clean_train(train, later, holdout, params)
    clean_path = resolve(params["data"]["cleaned_train"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False)
    print(f"train.csv sha256 {data_sha}\ncleaned rows: {len(clean)}")

    mlflow.set_tracking_uri(mf["tracking_uri"])
    client = MlflowClient()
    name = mf["registered_model_name"]
    if not args.force and client.search_model_versions(f"name='{name}'"):
        print(f"'{name}' is already registered. v1 exists; re-run with --force to add "
              "another version (Stage 4 expects the retrained model to be version 2).")
        return 1
    mlflow.set_experiment(mf["experiment_name"])

    common = {"data_sha256": data_sha, "n_rows_clean": len(clean),
              "val_size": mcfg["val_size"], "cv_folds": mcfg["cv_folds"],
              "random_state": mcfg["random_state"]}

    # 1. one run per candidate setting
    results, best = compare_candidates(clean, params)
    for r in results:
        with mlflow.start_run(run_name=r["name"]):
            mlflow.set_tags({"stage": "compare", "model_type": r["type"]})
            mlflow.log_params({**common, "model_type": r["type"],
                               **{f"hp_{k}": v for k, v in r["params"].items()}})
            mlflow.log_metrics({"cv_mae": r["cv_mae"], "cv_mae_std": r["cv_mae_std"],
                                "mae_val": r["mae_val"]})
        print(f"  {r['name']:<18} cv_mae={r['cv_mae']:.3f}  mae_val={r['mae_val']:.3f}")
    print(f"best setting: {best['name']}")

    # 2. refit the best on all cleaned rows -> v1, registered and @champion
    final = fit_final(best, clean, params)
    X, y = split_xy(clean, params)
    with mlflow.start_run(run_name=f"v1_{best['name']}") as run:
        mlflow.set_tags({"stage": "v1", "model_type": best["type"],
                         "trained_on": "cleaned train.csv (all rows)"})
        mlflow.log_params({**common, "model_type": best["type"], "selected_from": len(results),
                           **{f"hp_{k}": v for k, v in best["params"].items()}})
        mlflow.log_metrics({"cv_mae": best["cv_mae"], "mae_val": best["mae_val"],
                            "mae_train": mae(y, final.predict(X))})
        info = _log_model(mlflow, final, X, name)
        run_id = run.info.run_id

    version = getattr(info, "registered_model_version", None)
    if version is None:
        mine = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id]
        version = max(int(v.version) for v in mine)
    version = int(version)
    client.set_registered_model_alias(name, mf["champion_alias"], version)
    client.set_model_version_tag(name, str(version), "role", "v1")

    out = resolve(mcfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out)

    tracking = {"experiment_name": mf["experiment_name"], "best_run_id": run_id,
                "registered_model_name": name, "champion_version": version}
    update_findings_tracking(resolve(params["reports"]["findings"]), tracking)
    summary = {**tracking, "data_sha256": data_sha, "selected": best, "candidates": results}
    resolve("reports/train_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                     encoding="utf-8")

    print(f"registered {name} v{version} as @{mf['champion_alias']} (run {run_id})")
    print(f"saved {out.relative_to(ROOT)}; findings.yaml tracking block filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
