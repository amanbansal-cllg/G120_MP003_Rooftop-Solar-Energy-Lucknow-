# AmanBansal — owned files

Work in these canonical locations.

| Status | File or folder | Responsibility |
| --- | --- | --- |
| To create | `.github/workflows/ci.yml` | Push-triggered test workflow |
| To create | `k8s/deployment.yaml` | Three-replica `mp003-api` deployment |
| To create | `k8s/service.yaml` | `mp003-svc` service manifest |
| To create | `src/<package>/monitoring.py` | PSI, group MAE, retraining comparison |
| Shared | `reports/findings.yaml` | Record monitoring and promotion results |

Dependency: coordinate model, API, Docker and test integration before enabling
CI and creating the deployment manifests.
