# Rooftop Solar Energy Forecasting — Lucknow

**Group G120 · Problem MP003 · SCSE3040 MLOps Mini Project**

An end-to-end MLOps pipeline that predicts the daily energy output (kWh) of
rooftop solar panels for the **Lucknow Rooftop Solar Cooperative**, so that
members can schedule heavy loads for sunny hours and spot under-performing
panels.

![Python](https://img.shields.io/badge/python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458)
![DVC](https://img.shields.io/badge/data-DVC-945DD6)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![Stage](https://img.shields.io/badge/stage%201-complete-brightgreen)

---

## Contents

- [Overview](#overview)
- [Team](#team)
<!-- - [Project status](#project-status) -->
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Getting started](#getting-started)
- [Data quality findings](#data-quality-findings)
- [Cleaned-data contract](#cleaned-data-contract)
- [Configuration](#configuration)
- [Testing](#testing)
- [Service and deployment specification](#service-and-deployment-specification)
- [Monitoring and retraining plan](#monitoring-and-retraining-plan)
- [Contributing](#contributing)

---

## Overview

| | |
| --- | --- |
| **Client** | Lucknow Rooftop Solar Cooperative (fictional) |
| **Task** | Regression: predict `energy_kwh`, the energy produced in a day |
| **Metric** | MAE (Mean Absolute Error, in kWh) |
| **Training data** | `train.csv`, January–June 2026, 1,032 rows, intentionally messy |
| **Drift data** | `later.csv`, July–August 2026, 300 clean rows |
| **Holdout** | `holdout_features.csv`, September 2026, 250 rows, no target |

Full problem statement: [`PROBLEM.md`](PROBLEM.md). Course rules and marking:
[`STUDENT_HANDBOOK.md`](STUDENT_HANDBOOK.md).

## Team

| Enrolment no. | Member | Role | Owns |
| --- | --- | --- | --- |
| S24CSEU1793 | Arpit Kumar | Data lead | DVC, data checks, cleaning, data findings — [details](ARPITKUMAR_OWNED_FILES.md) |
| S24CSEU2215 | Rahul Nagilla | Model lead | Training, MLflow, champion model, holdout predictions — [details](Rahul_Nagilla_OWNED_FILES.md) |
| S24CSEU1783 | Alya Bathla | Service lead | FastAPI service, tests, Docker — [details](ALYA_OWNED_FILES.md) |
| S24CSEU1771 | Aman Bansal | Ops lead | CI, Kubernetes, logging, drift monitoring, retraining — [details](Aman_Bansal_OWNED_FILES.md) |

Hand-off order between roles: [`TEAM_OWNERSHIP.md`](TEAM_OWNERSHIP.md).

<!--
## Project status

| Stage | Scope | Owner | Status |
| --- | --- | --- | --- |
| 1 | Repository, DVC, seven data-quality checks, data part of `findings.yaml` | Arpit | ✅ Complete |
| 2 | Training with MLflow tracking, 3+ runs, registered `@champion` model | Rahul | ⏳ In progress |
| 3 | FastAPI service, Dockerfile, GitHub Actions CI, Kubernetes manifests, JSON logging | Alya, Aman | ⏳ In progress |
| 4 | PSI drift, per-group MAE, retrain v2, promotion decision, holdout predictions | Aman, Rahul | 🔜 Planned |
-->

## Pipeline

```mermaid
flowchart LR
    A[train.csv<br/>DVC-tracked] --> B[data_checks.py<br/>7 checks]
    B --> C[reports/findings.yaml]
    A --> D[cleaning.py]
    B --> D
    D --> E[train_clean.csv<br/>1,000 rows]
    E --> F[Training + MLflow<br/>Stage 2]
    F --> G[models/model.joblib<br/>@champion]
    G --> H[FastAPI /predict<br/>Stage 3]
    H --> I[Docker + Kubernetes<br/>Stage 3]
    J[later.csv] --> K[PSI drift + retrain v2<br/>Stage 4]
    G --> K
    K --> L[holdout_predictions.csv]
```

## Repository structure

```text
.
├── README.md                     # this file
├── PROBLEM.md                    # problem statement (from the course)
├── STUDENT_HANDBOOK.md           # course rules and marking
├── GROUP.md                      # group and members
├── TEAM_OWNERSHIP.md             # roles and hand-off order
├── *_OWNED_FILES.md              # each member's files and responsibilities
├── params.yaml                   # all paths, columns and thresholds
├── requirements.txt              # pinned dependencies
├── .dvc/  .dvcignore             # DVC configuration
├── data/
│   └── raw/
│       ├── train.csv.dvc         # DVC pointer (CSV itself is not in git)
│       ├── later.csv.dvc         # DVC pointer
│       └── holdout_features.csv  # committed directly (no target)
├── src/
│   └── solar_ml/
│       ├── __init__.py
│       ├── config.py             # loads params.yaml and data files
│       ├── data_checks.py        # detects the seven data problems
│       └── cleaning.py           # fixes them, writes the cleaned dataset
├── tests/
│   ├── conftest.py
│   └── test_data_checks.py       # 6 tests, no CSVs needed
└── reports/
    ├── data_checks.json          # raw output of data_checks.py
    └── findings.yaml             # the form read by the marking tool
```

Still to come from later stages: `app/main.py`, `Dockerfile`,
`.dockerignore`, `.github/workflows/ci.yml`, `k8s/deployment.yaml`,
`k8s/service.yaml`, `mlflow.db`, `models/model.joblib` and
`predictions/holdout_predictions.csv`.

## Dataset

| Column | Unit | Description |
| --- | --- | --- |
| `record_id` | text | Unique id of the record |
| `date` | YYYY-MM-DD | Date of the record |
| `panel_area_m2` | m² | Panel area |
| `sunshine_hours` | hours | Hours of bright sunshine |
| `temp_c` | °C | Maximum temperature |
| `dust_days` | days | Days since the panels were last cleaned |
| `panel_age_yr` | years | Age of the panels |
| `cloud_cover_pct` | % | Cloud cover at noon |
| `roof_type` | text | `flat`, `sloped` or `tin` |
| `area` | text | `Old City`, `New Colony` or `Industrial` |
| `energy_kwh` | kWh | **Target**: energy produced in the day |

The raw CSVs are versioned with **DVC**, not git. `train.csv.dvc` records the
MD5 of the original file (`d8d15d2aea399d09c28acaeb7f52332a`); `dvc status`
confirms a local copy is unchanged. Never open and re-save `train.csv` in
Excel — it changes the hash.

## Getting started

### Prerequisites

- Python 3.12 (3.10+ works for the data code)
- Git, and DVC for checking the data files

### Setup

```bash
git clone https://github.com/arpit-k16/G120_MP003_Rooftop-Solar-Energy-Lucknow-.git
cd G120_MP003_Rooftop-Solar-Energy-Lucknow-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

No DVC remote is configured, so copy `train.csv` and `later.csv` from the
course zip into `data/raw/`, then verify them against the committed hashes:

```bash
dvc status        # "Data and pipelines are up to date." means the files match
```

### Run the data checks and cleaning

```bash
export PYTHONPATH=src                       # Windows PowerShell: $env:PYTHONPATH="src"
python -m solar_ml.data_checks              # prints the 7 results, writes reports/data_checks.json
python -m solar_ml.cleaning                 # writes data/processed/train_clean.csv (1,000 rows)
```

## Data quality findings

Every finding below is produced by `src/solar_ml/data_checks.py` and recorded in
[`reports/findings.yaml`](reports/findings.yaml). Counts are on `train.csv` as
given (1,032 rows).

| Problem | Present | Finding | How it was detected |
| --- | :---: | --- | --- |
| Leaking column | No | — | Every feature exists in the holdout file; max \|correlation\| with the target is 0.19 |
| Unit mix-up | Yes | `panel_area_m2` in `New Colony` is in cm², factor **10000** | Group median ≈ 358,000 vs ≈ 33.5 elsewhere; snapped to the exact m² → cm² conversion |
| Sentinel value | Yes | `sunshine_hours` = **999** in **63** rows | `value_counts()` against known sentinels; real range is 2–11 h |
| Duplicate rows | Yes | **32** extra copies | `df.duplicated().sum()` |
| Stuck sensor | No | — | Sorted by date, no value repeats across 7+ days (longest run: 4 rows in one day) |
| Target typos | Yes | **13** rows of `energy_kwh` typed 10× too big | Largest gap in the sorted target: 49.78 → 131.2 |
| Spelling variants | Yes | `roof_type`, **3** true levels | 18 raw spellings normalised to `flat`, `sloped`, `tin` (as in `later.csv`) |

## Cleaned-data contract

`src/solar_ml/cleaning.py` applies the fixes in this order and writes
`data/processed/train_clean.csv` (not committed; regenerate it with the
command above):

1. Drop duplicate rows (1,032 → 1,000).
2. Map `roof_type` spellings to `flat` / `sloped` / `tin`.
3. Divide New Colony's `panel_area_m2` by 10000.
4. Replace `sunshine_hours` = 999 with the median of the real readings (6.2 h).
5. Divide the typo targets by 10.

**For model training:**

| | Columns |
| --- | --- |
| Numeric features | `panel_area_m2`, `sunshine_hours`, `temp_c`, `dust_days`, `panel_age_yr`, `cloud_cover_pct` |
| Categorical features | `roof_type`, `area` |
| Target | `energy_kwh` |
| Not features | `record_id`, `date` |

MLflow runs must log `data_sha256`, the SHA-256 of the original `train.csv`:
`5c46ec426e4f50f0469d970dee5d671889d90f06b97538f99b5f0221c8f8c3a5`
(`solar_ml.config.file_sha256()` computes it).

## Configuration

All settings live in [`params.yaml`](params.yaml) and are read with
`yaml.safe_load` in `src/solar_ml/config.py`:

| Section | Holds |
| --- | --- |
| `data` | Paths to the raw, holdout and cleaned files |
| `columns` | Id, date, target, group, numeric and categorical columns |
| `data_checks` | Thresholds: sentinel candidates, unit factors, minimum stuck run, leak correlation, typo gap |
| `reports` | Output path for the data-check results |

Later stages add their own sections (training, evaluation, monitoring) to the
same file.

## Testing

```bash
pytest -q
```

The suite builds small synthetic datasets inside each test, so it runs without
the CSVs (as CI requires). It covers all seven detectors, including the two
problems absent from our data, and an end-to-end cleaning run.

## Service and deployment specification

To be implemented in Stage 3, from [`PROBLEM.md`](PROBLEM.md):

| Item | Requirement |
| --- | --- |
| `POST /predict` | Body has exactly: `panel_area_m2`, `sunshine_hours`, `temp_c`, `dust_days`, `panel_age_yr`, `cloud_cover_pct`, `roof_type`, `area` |
| Validation | `panel_age_yr` ≤ 30 and `temp_c` ≥ 6, enforced in the pydantic schema (HTTP 422 otherwise) |
| `GET /health` | Returns `{"status": "ok"}` |
| Docker image | `mp003-api:1.0`, `python:3.12-slim`, port 8000 |
| Kubernetes | Deployment `mp003-api` (3 replicas), Service `mp003-svc` port 80 → 8000 |
| Probes | Readiness and liveness on `/health`; liveness waits longer |

## Monitoring and retraining plan

To be implemented in Stage 4:

1. PSI (10 bins) for every numeric column, cleaned `train.csv` vs `later.csv`.
2. v1's MAE on `later.csv` for each `area`.
3. Retrain v2 on cleaned train + the first 70% of `later.csv` (by date); compare on the last 30%.
4. Promote v2 to `@champion` only if its MAE is at least 15% lower than v1's.
5. Predict `holdout_features.csv` with the champion → `predictions/holdout_predictions.csv`.

## Contributing

- Every member commits their own work from their **own GitHub account**.
- Work in the files listed in your `*_OWNED_FILES.md`; ask for review before
  changing another member's files.
- Never commit raw CSVs (except `holdout_features.csv`) or anything in `data/processed/`.
- Keep every library in `requirements.txt` pinned with `==`.
- Run `pytest` before pushing.
- Pull before you start (`git pull --rebase origin main`) to avoid conflicts.

---

*Academic project for SCSE3040 MLOps · Faculty: Pulkit Dwivedi · Batch 59.
The client and data are fictional.*
