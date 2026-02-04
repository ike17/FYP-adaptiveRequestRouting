# Intelligent Task Scheduler for Distributed RAG Workloads on Kubernetes

## Project Overview

This project implements and evaluates ML-based Kubernetes schedulers for optimizing Retrieval-Augmented Generation (RAG) workloads on heterogeneous clusters.

**Research Question:** Can a Detection-Augmented Bandit (DAB) scheduler mitigate performance degradation in piecewise-stationary heterogeneous cluster environments better than Static ML or Default Kubernetes scheduling?

## Directory Structure

```
fyp-rag-scheduler/
├── README.md                          # This file
├── generation-service/                # GPU-bound LLM inference service
│   ├── Dockerfile
│   └── pull-model.sh
├── rag-app/                           # CPU-bound retrieval + API service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   └── vectordb.py
├── ml-schedulers/
│   ├── static/                        # Static ML Scheduler (Random Forest)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── generate_dataset.py
│   │   ├── train_model.py
│   │   └── static_scheduler.py
│   └── bandit/                        # Adaptive Bandit Scheduler (Thompson Sampling)
│       ├── Dockerfile
│       ├── requirements.txt
│       └── bandit_scheduler.py
├── k8s/                               # Kubernetes manifests
│   ├── baseline/
│   │   ├── 01-uninformed-default.yaml
│   │   ├── 02-informed-default.yaml
│   │   └── 03-manual-optimal.yaml
│   ├── static-scheduler/
│   │   ├── rbac.yaml
│   │   ├── scheduler-deployment.yaml
│   │   └── rag-deployment.yaml
│   └── bandit-scheduler/
│       ├── rbac.yaml
│       ├── scheduler-deployment.yaml
│       └── rag-deployment.yaml
├── benchmarks/                        # Evaluation scripts
│   ├── requirements.txt
│   ├── run_experiments.py
│   └── analyze_results.py
├── scripts/                           # Helper scripts
│   ├── setup_cluster.sh
│   ├── build_all.sh
│   └── run_all_experiments.sh
└── results/                           # Output directory (gitignored)
    └── .gitkeep
```

## Prerequisites

- Windows 11 with WSL2 (Ubuntu 22.04)
- Docker Desktop with WSL2 backend
- NVIDIA GPU with drivers installed in WSL2
- 32GB RAM recommended (16GB minimum)
- ~20GB disk space

## Quick Start

### Step 1: Install Dependencies

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Verify NVIDIA driver in WSL2
nvidia-smi
```

### Step 2: Setup Cluster

```bash
cd fyp-rag-scheduler
chmod +x scripts/*.sh
./scripts/setup_cluster.sh
```

### Step 3: Build All Images

```bash
./scripts/build_all.sh
```

### Step 4: Train Static ML Model

```bash
cd ml-schedulers/static
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
cd ../..
```

### Step 5: Run Experiments

```bash
./scripts/run_all_experiments.sh
```

### Step 6: Analyze Results

```bash
cd benchmarks
pip install -r requirements.txt
python analyze_results.py
```

## Experiment Configurations

| Configuration | Scheduler | Description |
|--------------|-----------|-------------|
| baseline-uninformed | default | No hints, random placement |
| baseline-informed | default | GPU resource requests specified |
| baseline-optimal | default | Manual nodeSelector (theoretical best) |
| static-ml | static-scheduler | Random Forest prediction |
| bandit-adaptive | bandit-scheduler | Thompson Sampling with regime detection |

## Expected Results

- **Static ML:** ~20-25% latency improvement over uninformed baseline
- **Bandit (normal):** Similar to Static ML after exploration phase
- **Bandit (shock):** Recovers to CPU-optimal after detecting GPU degradation

## Shock Scenario

At query 50 (out of 100), we simulate GPU degradation:
- Before shock: GPU=0.5s, CPU=10s → Optimal: GPU
- After shock: GPU=15s, CPU=10s → Optimal: CPU

The Bandit scheduler detects this regime change and adapts. The Static scheduler continues failing.

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues.

## License

MIT License - See LICENSE file
