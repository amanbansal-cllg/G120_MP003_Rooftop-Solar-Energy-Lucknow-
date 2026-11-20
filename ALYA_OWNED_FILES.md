# Alya Bathla — owned files

Work directly in these canonical repository locations.

| File or folder | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI `GET /health` and `POST /predict` service |
| `tests/test_api.py` | TestClient API and validation tests |
| `tests/test_cleaning.py` | Cleaning-related tests, if needed for service integration |
| `tests/test_model.py` | Model-loading and prediction tests |
| `Dockerfile` | Python 3.12-slim API container |
| `.dockerignore` | Docker build exclusions |
| `requirements.txt` | Coordinate service dependencies with the model lead |

## Required API rules

- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` accepts exactly: `panel_area_m2`, `sunshine_hours`,
  `temp_c`, `dust_days`, `panel_age_yr`, `cloud_cover_pct`, `roof_type`, and
  `area`.
- Validate requests in the Pydantic schema: `panel_age_yr <= 30` and
  `temp_c >= 6`, producing HTTP 422 for invalid values.

Commit only the files Alya changes, using Alya's own GitHub account and Git
identity.
