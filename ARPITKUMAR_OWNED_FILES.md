# Arpit Kumar — owned files

Work directly in these canonical repository locations.

| File or folder | Responsibility |
| --- | --- |
| `data/raw/train.csv.dvc` | DVC metadata for the original training data |
| `data/raw/later.csv.dvc` | DVC metadata for later-arriving data |
| `data/raw/holdout_features.csv` | Holdout features, committed directly (handbook section 5) |
| `src/solar_ml/__init__.py` | Python package initializer |
| `src/solar_ml/config.py` | Reads `params.yaml` and the raw data files |
| `src/solar_ml/data_checks.py` | Reproducible checks for all seven data problems |
| `src/solar_ml/cleaning.py` | Cleaning functions and feature preparation |
| `tests/test_data_checks.py` | Tests for the data checks and cleaning (no CSVs needed) |
| `reports/data_checks.json` | Output of `data_checks`, the evidence for findings |
| `reports/findings.yaml` | Complete the data-issues section with computed evidence |

## Stage 1 responsibilities

1. Do not alter the raw CSV files or open and re-save `train.csv` in Excel.
2. Detect each of the seven data problems in `train.csv` using code.
3. Record whether each problem is present and fill the required details in
   `reports/findings.yaml`.
4. Keep the raw CSV files ignored by Git; commit only their `.dvc` metadata.

Commit only the files Arpit changes, using Arpit's own GitHub account and Git
identity.
