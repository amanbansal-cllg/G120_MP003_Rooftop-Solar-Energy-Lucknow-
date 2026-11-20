# G120 — MP003: Rooftop Solar Energy (Lucknow)

SCSE3040 MLOps mini project for the Lucknow Rooftop Solar Cooperative.

## Team

| Enrolment no. | Member | Primary ownership |
| --- | --- | --- |
| S24CSEU1793 | ARPITKUMAR | Data quality and DVC |
| S24CSEU1771 | AmanBansal | Delivery and operations |
| S24CSEU1783 | AlyaBathla | FastAPI service and tests |
| S24CSEU2215 | RAHULNAGILLA | Modelling and MLflow |

Problem: **MP003 — Rooftop Solar Energy (Lucknow)**

## Working layout

Each member works directly in the course-required locations:

```
data/raw/                 # DVC-tracked source datasets (do not commit CSVs)
src/                      # data and modelling code
app/                      # prediction service
tests/                    # automated tests
models/                   # champion model for the service
reports/                  # completed findings form
predictions/              # holdout predictions
k8s/                      # Kubernetes manifests
.github/workflows/        # continuous integration
```

## Data checks and cleaning (Stage 1)

```bash
pip install -r requirements.txt
dvc pull                                   # or copy the CSVs from the zip into data/raw/
PYTHONPATH=src python -m solar_ml.data_checks   # finds the seven data problems -> reports/data_checks.json
PYTHONPATH=src python -m solar_ml.cleaning      # writes data/processed/train_clean.csv
pytest
```

On Windows PowerShell, set the path first with `$env:PYTHONPATH="src"`.

Problems found in `train.csv` (also in `reports/findings.yaml`):

| Problem | Present | Details |
| --- | --- | --- |
| Leaking column | No | every feature is also in `holdout_features.csv`; none tracks the target |
| Unit mix-up | Yes | `panel_area_m2` for `New Colony` is in cm² (x10000) |
| Sentinel value | Yes | `sunshine_hours` = 999 in 63 rows |
| Duplicate rows | Yes | 32 extra copies |
| Stuck sensor | No | no column repeats one value over a run of days |
| Target typos | Yes | 13 rows of `energy_kwh` typed 10x too big |
| Spelling variants | Yes | `roof_type` has 3 real levels: flat, sloped, tin |

Cleaned data: 1000 rows. All settings are in `params.yaml`.

Read `PROBLEM.md` and `STUDENT_HANDBOOK.md` before contributing. Every member
should commit their own work from their own GitHub account.
