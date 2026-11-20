# ARPITKUMAR — owned files

Work in these canonical locations; do not duplicate raw data in this folder.

| Status | File or folder | Responsibility |
| --- | --- | --- |
| Ready | `data/raw/train.csv.dvc` | DVC metadata for the original training data |
| Ready | `data/raw/later.csv.dvc` | DVC metadata for later-arriving data |
| Ready | `data/raw/holdout_features.csv.dvc` | DVC metadata for holdout features |
| Ready | `reports/findings.yaml` | Complete the data-quality findings |
| To create | `src/<package>/data_checks.py` | Reproducible checks for the seven data problems |
| To create | `src/<package>/cleaning.py` | Cleaning functions and feature contract |

First-submission check: confirm the raw CSVs are ignored by Git and only the
`.dvc` files are committed.
