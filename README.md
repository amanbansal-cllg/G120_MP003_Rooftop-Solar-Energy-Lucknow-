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

Each member has an ownership note in `team/`.  The shared deliverables are
kept in the course-required locations:

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

Read `PROBLEM.md` and `STUDENT_HANDBOOK.md` before contributing. Every member
should commit their own work from their own GitHub account.
