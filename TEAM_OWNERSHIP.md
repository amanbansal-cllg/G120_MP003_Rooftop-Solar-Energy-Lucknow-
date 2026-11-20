# Team ownership and hand-offs

This is an ownership guide. Course-required deliverables stay in their
canonical repository paths.

| Owner | Workspace | Expected hand-off |
| --- | --- | --- |
| ARPITKUMAR | `data/`, `src/` data checks | DVC metadata, reproducible data checks, cleaned-data contract, data portion of findings |
| AmanBansal | `.github/`, `k8s/`, monitoring code | green CI, manifests, logs, drift/retraining evidence |
| AlyaBathla | `app/`, `tests/`, `Dockerfile` | validated API, TestClient tests, container instructions |
| RAHULNAGILLA | `src/` modelling, `models/`, `predictions/` | tracked runs, registered champion model, holdout predictions |

## Shared hand-off order

1. Data lead documents the cleaned feature and target contract.
2. Model lead supplies the serialized champion model and inference expectations.
3. Service lead integrates that model behind the API and tests it.
4. Ops lead validates the container, CI, Kubernetes manifests, monitoring and retraining outputs.

Use pull requests or peer review before merging work that crosses another
member's ownership boundary.
