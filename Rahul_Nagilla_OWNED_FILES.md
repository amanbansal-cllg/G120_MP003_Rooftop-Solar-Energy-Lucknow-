# RAHULNAGILLA — owned files

Work in these canonical locations.

| Status | File or folder | Responsibility |
| --- | --- | --- |
| To create | `params.yaml` | Training and evaluation settings |
| To create | `src/<package>/train.py` | Training and MLflow runs |
| To create | `src/<package>/evaluate.py` | MAE evaluation and model comparison |
| To create | `models/model.joblib` | Service-ready champion model |
| To create | `mlflow.db` | SQLite MLflow tracking database |
| To create | `predictions/holdout_predictions.csv` | Final holdout predictions |

Dependency: use the cleaned-data contract supplied by the data lead.
