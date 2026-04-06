# Troubleshooting

Common setup and runtime issues for the FYP RAG scheduler project.

## 1) k3d cluster fails to create

- Verify Docker Desktop is running.
- Delete any existing cluster and retry:
  ```bash
  k3d cluster delete fyp
  ./scripts/setup_cluster.sh
  ```
- Check Docker has enough resources (4+ CPUs, 8+ GB RAM allocated in Docker Desktop settings).

## 2) Pods stuck in `Pending`

- Run `kubectl describe pod <pod-name>` and look at the Events section.
- Confirm node labels exist:
  ```bash
  kubectl get nodes --show-labels | grep node_type
  ```
  Expected: `node_type=cpu_standard` on server-0, `node_type=gpu_accelerated` on agent-0.
- Ensure the custom scheduler pod is Running before deploying workloads with `schedulerName: k3d-scheduler`.
- Ensure RBAC was applied: `kubectl apply -f k8s/bandit-scheduler/rbac-adaptive.yaml`

## 3) RAG service not reachable

- Verify services exist: `kubectl get svc`
- Check pod readiness: `kubectl get pods -o wide`
- On Windows, always use `--port-forward` with `run_experiments.py`:
  ```bash
  python benchmarks/run_experiments.py --config <name> --port-forward --skip-wait ...
  ```

## 4) First query is very slow (~60-100s)

Expected — the generation-service pulls and loads `gemma:2b` on first startup via the postStart hook. Subsequent queries are 5-15s. The experiment runner waits 60s before sending queries; increase this if the model is still loading:

```bash
# In run_all.sh, change:
sleep 60
# to:
sleep 120
```

## 5) Adaptive scheduler does not react to shock

- Verify adaptive manifests are deployed:
  - `k8s/bandit-scheduler/scheduler-deployment-adaptive.yaml`
  - `k8s/bandit-scheduler/rag-deployment-adaptive.yaml`
- Confirm environment variables in the scheduler deployment:
  - `SIMULATE_REWARDS=true` (required for k3d CPU-only clusters)
  - `SHOCK_ENABLED=true`
  - `SHOCK_QUERY=50`

## 6) Metrics endpoint returns empty history

- Send at least one query to `/query` before checking `/metrics`.
- Clear between runs with `DELETE /metrics`:
  ```bash
  curl -X DELETE http://localhost:8080/metrics
  ```

## 7) Port-forward drops during benchmark run

k3d port-forwarding is unreliable on Windows/WSL2. The `PortForwardManager` in `run_experiments.py` auto-restarts dead tunnels — expect many `[Restarted] kubectl port-forward` messages, this is normal.

If a query still fails after 3 retries it is recorded as `success: false`. This is expected for the baseline experiment where slow inference causes the rag-app liveness probe to kill the pod.

## 8) Images not found in cluster (ErrImageNeverPull)

k3d nodes do not share the host Docker daemon. After any `docker build`, reimport:

```bash
k3d image import <image-name> -c fyp
# or rebuild everything:
./scripts/build_all.sh
```

## 9) UnicodeEncodeError on Windows

The Windows console uses cp1252 by default. Set UTF-8 before running Python scripts:

```bash
# Git Bash / WSL
export PYTHONIOENCODING=utf-8

# PowerShell
$env:PYTHONIOENCODING="utf-8"
```

## 10) Clean reset between experiments

`run_all.sh` deletes deployments and services before each experiment automatically. For a manual reset:

```bash
kubectl delete deployment --all
kubectl delete service generation-service rag-app-service --ignore-not-found
```

To destroy the entire cluster:

```bash
k3d cluster delete fyp
```
