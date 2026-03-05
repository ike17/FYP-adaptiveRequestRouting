# QUICKSTART GUIDE

## Prerequisites

| Tool | Install |
|---|---|
| Docker Desktop | [docker.com](https://www.docker.com/products/docker-desktop/) |
| k3d | `choco install k3d` / `brew install k3d` / [k3d.io](https://k3d.io) |
| kubectl | `choco install kubernetes-cli` / `brew install kubectl` |
| Python 3.9+ | [python.org](https://www.python.org) |

No GPU required. The cluster simulates hardware heterogeneity via Docker CPU quotas.

---

## Step 1 — Create the cluster

```bash
./scripts/setup_cluster.sh
```

This creates a k3d cluster named `fyp` with:
- `k3d-fyp-server-0` — labeled `node_type=cpu_standard`, capped at **1 CPU**
- `k3d-fyp-agent-0` — labeled `node_type=gpu_accelerated`, capped at **4 CPUs**

Verify nodes are ready:

```bash
kubectl get nodes --show-labels
```

---

## Step 2 — Train the static ML model

```bash
cd ml-schedulers/static
python generate_dataset.py    # creates training_data.csv (2000 samples)
python train_model.py         # trains Random Forest, saves static_scheduler_model.pkl
cd ../..
```

> **Windows note:** If you get a `UnicodeEncodeError`, prefix with `PYTHONIOENCODING=utf-8`

Expected output from `train_model.py`:
```
Test Set Accuracy: 0.94
Model saved: static_scheduler_model.pkl
```

---

## Step 3 — Build Docker images

```bash
./scripts/build_all.sh
```

Builds and imports into k3d:
- `generation-service:v1` — Ollama + gemma:2b
- `rag-app:v1` — FastAPI + ChromaDB
- `static-scheduler:v1` — Random Forest scheduler
- `bandit-scheduler:v1` — Thompson Sampling scheduler
- `bandit-scheduler:v2-adaptive` — Adaptive variant with regime detection

Verify:

```bash
docker images | grep -E "(generation-service|rag-app|scheduler)"
```

---

## Step 4 — Install Python dependencies

```bash
pip install requests pandas numpy matplotlib seaborn scipy tqdm
```

> Do not use `pip install -r benchmarks/requirements.txt` on Python 3.12+ — the pinned scipy version has no pre-built wheel and will fail to compile.

---

## Step 5 — Apply RBAC

```bash
kubectl apply -f k8s/static-scheduler/rbac.yaml
kubectl apply -f k8s/bandit-scheduler/rbac-adaptive.yaml
```

---

## Step 6 — Run all experiments

```bash
bash run_all.sh
```

Runs four experiments sequentially (each 100 queries, ~20-30 min per experiment):

1. `baseline-uninformed` — default scheduler, no placement hints
2. `static-ml` — Random Forest scheduler
3. `bandit` — Thompson Sampling bandit
4. `bandit-adaptive` — Bandit with sliding-window regime detection

Results are saved to `results/` as `{experiment}_results.json` and `{experiment}_summary.json`.

**To run a single experiment manually:**

```bash
# Apply RBAC (if not done already)
kubectl apply -f k8s/static-scheduler/rbac.yaml

# Deploy scheduler + RAG app
kubectl apply -f k8s/static-scheduler/scheduler-deployment.yaml
kubectl apply -f k8s/static-scheduler/rag-deployment.yaml

# Wait for all pods to be Running
kubectl get pods -w

# Wait 60 seconds for the model to load, then run queries
python benchmarks/run_experiments.py \
  --config static-ml \
  --queries 100 \
  --delay 0.5 \
  --output results \
  --port-forward \
  --skip-wait

# Cleanup before next experiment
kubectl delete deployment --all
kubectl delete service --all
```

---

## Step 7 — Analyse results

```bash
python benchmarks/analyze_results.py --input results --output results/analysis
```

Output in `results/analysis/`:

| File | Contents |
|---|---|
| `latency_comparison.png` | Mean latency bar chart |
| `percentile_comparison.png` | P50 / P95 / P99 |
| `latency_over_time.png` | Per-query latency with shock at Q50 |
| `post_shock_convergence.png` | Bandit adaptation post-shock |
| `statistical_tests.json` | Welch's t-test, Bonferroni correction, Cohen's d |
| `comparison_table.csv` | Summary table |

---

## Quick reference

```bash
# Check cluster nodes
kubectl get nodes -o wide

# Watch pods start
kubectl get pods -w

# Check pod placement (which node)
kubectl get pods -o wide

# Check scheduler logs
kubectl logs -l app=static-scheduler
kubectl logs -l app=k3d-scheduler

# View pod details / events
kubectl describe pod <pod-name>

# Manual port-forward for testing
kubectl port-forward svc/rag-app-service 8080:8000

# Test health endpoint
curl http://localhost:8080/health

# Send a test query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Kubernetes?", "include_context": true}'

# Cleanup all deployments and services
kubectl delete deployment --all
kubectl delete service --all

# Destroy the cluster entirely
k3d cluster delete fyp
```

---

## Troubleshooting

**Port-forward dies on Windows**

Expected — k3d port-forwarding is unstable on Windows. Use `--port-forward` flag with `run_experiments.py` which auto-restarts the tunnel. You will see many `[Restarted] kubectl port-forward` messages; this is normal.

**Pods stuck in Pending**

```bash
kubectl describe pod <pod-name>
# Look for: "0/2 nodes are available" or scheduler errors
# Fix: ensure RBAC was applied and scheduler pod is Running
```

**generation-service keeps restarting**

The liveness probe fires when inference is slow (usually means the pod landed on the wrong node). With smart schedulers this should not happen. With baseline, occasional restarts are expected.

**Images not found after rebuild**

k3d nodes don't share the host Docker daemon. Always re-import after rebuilding:

```bash
k3d image import <image-name> -c fyp
```

**Model not loaded after 60s wait**

The `ollama pull gemma:2b` postStart hook may take longer than 60s on a slow network. Check:

```bash
kubectl logs <generation-pod-name>
```

If the model is still downloading, increase the wait in `run_all.sh` (`sleep 60` → `sleep 120`).
