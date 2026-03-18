# Intelligent Task Scheduler for Distributed RAG Workloads on Kubernetes

## Project Overview

This project implements and evaluates ML-based Kubernetes schedulers for optimising Retrieval-Augmented Generation (RAG) workloads on heterogeneous clusters.

**Research Question:** Can a Detection-Augmented Bandit (DAB) scheduler mitigate performance degradation in piecewise-stationary heterogeneous cluster environments better than Static ML or Default Kubernetes scheduling?

## Architecture

Two-stage RAG pipeline running on a 2-node k3d cluster:

- **generation-service** — Ollama + gemma:2b (LLM inference, compute-heavy)
- **rag-app** — FastAPI + ChromaDB (retrieval + orchestration, lighter)

The cluster simulates heterogeneous hardware via CPU quotas:

| Node | Label | CPU quota | Role |
|---|---|---|---|
| k3d-fyp-server-0 | `node_type=cpu_standard` | 1 CPU | Retrieval |
| k3d-fyp-agent-0 | `node_type=gpu_accelerated` | 4 CPUs | Generation |

## Directory Structure

```
fyp-rag-scheduler/
├── generation-service/        # Ollama LLM inference service
├── rag-app/                   # FastAPI retrieval + orchestration
├── ml-schedulers/
│   ├── static/                # Random Forest scheduler
│   └── bandit/                # Thompson Sampling scheduler (+ adaptive variant)
├── k8s/
│   ├── baseline/              # 01-uninformed-default.yaml
│   ├── static-scheduler/      # RBAC, scheduler deployment, RAG deployment
│   └── bandit-scheduler/      # RBAC, scheduler deployment, RAG deployment (+ adaptive)
├── benchmarks/
│   ├── run_experiments.py     # Query runner with managed port-forward
│   └── analyze_results.py     # Statistical analysis + chart generation
├── scripts/
│   ├── setup_cluster.sh       # Create k3d cluster + node labels + CPU quotas
│   └── build_all.sh           # Build Docker images + import into k3d
├── run_all.sh                 # Master experiment runner
└── results/                   # Output directory
```

## Prerequisites

- Docker Desktop (Windows) or Docker Engine (Linux)
- [k3d](https://k3d.io) — `choco install k3d` or `brew install k3d`
- kubectl — `choco install kubernetes-cli` or `brew install kubectl`
- Python 3.9+
- No GPU required — the cluster simulates hardware heterogeneity via CPU quotas

## Quick Start

### 1. Setup cluster

```bash
./scripts/setup_cluster.sh
```

Creates a k3d cluster named `fyp` with one server node (1 CPU) and one agent node (4 CPUs), labels both nodes, and applies Docker CPU quotas to simulate heterogeneous hardware.

Verify:

```bash
kubectl get nodes --show-labels | grep node_type
```

### 2. Train the static ML model

```bash
cd ml-schedulers/static
python generate_dataset.py   # generates 2000 synthetic training samples
python train_model.py        # trains Random Forest, saves .pkl files
cd ../..
```

Expected output: `Test Set Accuracy: ~0.94`

> **Windows note:** If `generate_dataset.py` produces a UnicodeEncodeError, run with
> `PYTHONIOENCODING=utf-8 python generate_dataset.py`

### 3. Build and import Docker images

```bash
./scripts/build_all.sh
```

Builds five images and imports them directly into the k3d cluster (no registry needed):

- `generation-service:v1`
- `rag-app:v1`
- `static-scheduler:v1`
- `bandit-scheduler:v1`
- `bandit-scheduler:v2-adaptive`

### 4. Install benchmark dependencies

```bash
pip install requests pandas numpy matplotlib seaborn scipy tqdm
```

> **Note:** Do not use pinned versions from `benchmarks/requirements.txt` on Python 3.12+ — scipy has no pre-built wheel and requires a Fortran compiler to build from source.

### 5. Apply RBAC

```bash
kubectl apply -f k8s/static-scheduler/rbac.yaml
kubectl apply -f k8s/bandit-scheduler/rbac-adaptive.yaml
```

### 6. Run all experiments

```bash
bash run_all.sh
```

Runs four experiments in sequence (baseline-uninformed → static-ml → bandit → bandit-adaptive), 100 queries each with a 0.5s inter-query delay. Results are saved to `results/`.

### Run multiple iterations (Optional)

To easily assess statistical variations and obtain more robust evaluations over multiple runs (e.g. 10x), run:

```bash
bash run_multiple.sh
```

This runs the experiments 10 times and outputs individual results in `results/run_{1..10}/` and then performs aggregated statistical analyses (n=1000) inside `results/aggregated_analysis/`.

Alternatively, run a single experiment manually:

```bash
# Deploy
kubectl apply -f k8s/baseline/01-uninformed-default.yaml

# Wait for pods
kubectl get pods -w   # wait for 2/2 Running

# Run queries (manages its own port-forward)
python benchmarks/run_experiments.py \
  --config baseline-uninformed \
  --queries 100 \
  --delay 0.5 \
  --output results \
  --port-forward \
  --skip-wait

# Cleanup
kubectl delete deployment --all
kubectl delete service --all
```

### 7. Analyse results

```bash
python benchmarks/analyze_results.py --input results --output results/analysis
```

Produces in `results/analysis/`:

| File | Description |
|---|---|
| `latency_comparison.png` | Mean latency bar chart with std error bars |
| `percentile_comparison.png` | P50 / P95 / P99 comparison |
| `latency_over_time.png` | Per-query latency showing shock at Q50 |
| `post_shock_convergence.png` | Bandit convergence post-shock |
| `statistical_tests.json` | Welch's t-test + Bonferroni + Cohen's d |
| `comparison_table.csv` | Summary table |

## Experiment Configurations

| Config | Scheduler | Description |
|---|---|---|
| `baseline-uninformed` | default kube-scheduler | No placement hints — worst case |
| `static-ml` | static-scheduler | Random Forest prediction on node features |
| `bandit` | k3d-scheduler | Thompson Sampling Beta-Bernoulli bandit |
| `bandit-adaptive` | k3d-scheduler (adaptive) | Bandit + sliding-window regime detection |

## Actual Results (k3d CPU simulation)

| Scheduler | Successful | Mean (ms) | P50 (ms) | P95 (ms) | vs Baseline |
|---|---|---|---|---|---|
| baseline-uninformed | 20/100 | 19,376 | 17,971 | 31,848 | — |
| static-ml | 56/100 | 8,143 | 5,270 | 26,687 | **+58%** |
| bandit | 54/100 | 7,335 | 4,883 | 24,840 | **+62%** |
| bandit-adaptive | 36/100 | 8,162 | 4,673 | 26,409 | **+58%** |

All improvements are statistically significant (Welch's t-test, Bonferroni-corrected p < 0.007, Cohen's d > 1.6).

The lower success rates reflect liveness probe kills caused by slow LLM inference on the wrong node — itself a consequence of poor scheduling. The smart schedulers have higher success rates because they correctly place the generation-service on the 4-CPU agent node.

### Pre/Post shock (Q50 boundary)

| Scheduler | Pre-shock mean | Post-shock mean |
|---|---|---|
| static-ml | 7,830ms | 9,178ms (+17% — cannot adapt) |
| bandit | 7,462ms | 6,837ms (−8% — adapted after shock) |

## Shock Scenario

At query 50, the bandit scheduler's reward function flips to simulate GPU degradation. The Thompson Sampling algorithm detects the distribution shift via a sliding window (W=20, threshold=0.4) and re-explores node assignments.

To adjust the shock point, edit `SIMULATE_REWARDS` / shock counter logic in [ml-schedulers/bandit/bandit_scheduler.py](ml-schedulers/bandit/bandit_scheduler.py).

## Troubleshooting

**Port-forward dies frequently on Windows**

k3d port-forwarding is unstable on Windows/WSL2. The `run_experiments.py` `--port-forward` flag uses `PortForwardManager` to auto-restart the tunnel on each connection failure. Expect many restarts — queries still succeed via the retry mechanism.

**UnicodeEncodeError on Windows console**

Set the console encoding before running Python scripts:

```bash
set PYTHONIOENCODING=utf-8   # Windows CMD
$env:PYTHONIOENCODING="utf-8"  # PowerShell
export PYTHONIOENCODING=utf-8  # Git Bash / WSL
```

**Pod keeps restarting (CrashLoopBackOff)**

The generation-service liveness probe fires if inference is slow (model on wrong node). If using the baseline config, this is expected — the default scheduler may place the container on the 1-CPU node.

```bash
kubectl describe pod <pod-name>   # check Events section
kubectl logs <pod-name>           # check container logs
```

**Pods stuck in Pending**

```bash
kubectl describe pod <pod-name>   # look for "Insufficient cpu" or scheduler errors
kubectl logs -l app=static-scheduler   # check scheduler decisions
```

**Images not found (ErrImageNeverPull)**

Images must be imported into k3d after every `docker build`:

```bash
k3d image import generation-service:v1 rag-app:v1 \
  static-scheduler:v1 bandit-scheduler:v1 bandit-scheduler:v2-adaptive \
  -c fyp
```

## License

MIT License — see [LICENSE](LICENSE)
