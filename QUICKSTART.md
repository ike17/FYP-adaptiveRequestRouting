# QUICKSTART GUIDE

## Complete Step-by-Step Setup and Execution

This guide walks you through every step to get the project running from scratch.

---

## Prerequisites Installation (One-time Setup)

### Step 1: Install Docker Desktop

```bash
# For Windows: Download and install Docker Desktop from docker.com
# Enable WSL2 backend in Docker Desktop settings

# For Linux:
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER
# Log out and back in
```

### Step 2: Install kubectl

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify
kubectl version --client
```

### Step 3: Install Minikube

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Verify
minikube version
```

### Step 4: Install Python 3.11+

```bash
# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv python3-pip

# Verify
python3 --version
```

### Step 5: Verify NVIDIA GPU (Optional but Recommended)

```bash
# Check if nvidia-smi works in WSL2
nvidia-smi

# If this shows your GPU, you're good!
# If not, the project will still work but LLM inference will be slower
```

---

## Project Setup

### Step 1: Clone/Copy the Project

```bash
# If you have the project as a zip:
unzip fyp-rag-scheduler.zip
cd fyp-rag-scheduler

# Or if copying:
cp -r /path/to/fyp-rag-scheduler ~/fyp-rag-scheduler
cd ~/fyp-rag-scheduler
```

### Step 2: Make Scripts Executable

```bash
chmod +x scripts/*.sh
```

### Step 3: Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

# Install dependencies for ML training
pip install -r ml-schedulers/static/requirements.txt
pip install -r benchmarks/requirements.txt
```

---

## Phase 0: Cluster Setup

### Step 1: Start Minikube Cluster

```bash
./scripts/setup_cluster.sh
```

**What this does:**
- Starts a 2-node Minikube cluster
- Labels `minikube` as `cpu_standard`
- Labels `minikube-m02` as `gpu_accelerated`
- Installs NVIDIA device plugin (if GPU available)

**Expected output:**
```
[✓] Minikube cluster started
[✓] All nodes ready
[✓] minikube labeled as cpu_standard
[✓] minikube-m02 labeled as gpu_accelerated
```

### Step 2: Verify Cluster

```bash
kubectl get nodes --show-labels

# Expected:
# NAME           STATUS   ROLES           AGE   VERSION   LABELS
# minikube       Ready    control-plane   1m    v1.28.0   node_type=cpu_standard,...
# minikube-m02   Ready    <none>          1m    v1.28.0   node_type=gpu_accelerated,...
```

---

## Phase 1: Build Docker Images

### Step 1: Train the Static ML Model First

```bash
cd ml-schedulers/static

# Generate synthetic training data
python generate_dataset.py

# Expected output:
# Generated 2000 training samples
# Dataset saved to: training_data.csv

# Train Random Forest model
python train_model.py

# Expected output:
# Test Set Accuracy: 0.94xx
# ✓ Model saved: static_scheduler_model.pkl
# ✓ Feature names saved: feature_names.pkl

cd ../..
```

### Step 2: Build All Docker Images

```bash
./scripts/build_all.sh
```

**What this does:**
- Configures Docker to use Minikube's daemon
- Builds `generation-service:v1` (Ollama + gemma:2b)
- Builds `rag-app:v1` (FastAPI + ChromaDB)
- Builds `static-scheduler:v1` (Random Forest scheduler)
- Builds `bandit-scheduler:v1` (Thompson Sampling scheduler)

**Expected output:**
```
[✓] generation-service:v1 built
[✓] rag-app:v1 built
[✓] static-scheduler:v1 built
[✓] bandit-scheduler:v1 built
```

### Step 3: Verify Images

```bash
# Make sure you're using Minikube's Docker
eval $(minikube docker-env)

docker images | grep -E "(generation|rag-app|scheduler)"
```

---

## Phase 2: Test Deployment (Manual)

Before running automated experiments, test that everything works:

### Step 1: Deploy Baseline Optimal Configuration

```bash
kubectl apply -f k8s/baseline/03-manual-optimal.yaml
```

### Step 2: Watch Pods Start

```bash
# In a separate terminal:
kubectl get pods -w

# Wait until you see:
# NAME                                    READY   STATUS    RESTARTS   AGE
# generation-deployment-xxx               1/1     Running   0          2m
# rag-deployment-xxx                      1/1     Running   0          2m
```

**Note:** The generation pod takes 2-5 minutes to start because it downloads the LLM model.

### Step 3: Check Pod Placement

```bash
kubectl get pods -o wide

# Expected:
# generation-deployment-xxx   minikube-m02   (GPU node)
# rag-deployment-xxx          minikube       (CPU node)
```

### Step 4: Test RAG Endpoint

```bash
# Get the service URL
RAG_URL=$(minikube service rag-app-service --url)
echo "RAG URL: $RAG_URL"

# Test health endpoint
curl $RAG_URL/health

# Expected: {"status":"healthy","service":"rag-app",...}

# Test a query (this will take 10-60 seconds first time)
curl -X POST "$RAG_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Kubernetes?", "include_context": true}'

# Expected: JSON with response, latency metrics
```

### Step 5: Check Metrics

```bash
curl $RAG_URL/metrics | python3 -m json.tool
```

### Step 6: Cleanup Test Deployment

```bash
kubectl delete -f k8s/baseline/03-manual-optimal.yaml
```

---

## Phase 3: Run Automated Experiments

### Option A: Run All Experiments (Recommended)

```bash
./scripts/run_all_experiments.sh
```

**What this does:**
1. Runs baseline-uninformed (100 queries)
2. Runs baseline-informed (100 queries)
3. Runs baseline-optimal (100 queries)
4. Runs static-ml scheduler (100 queries)
5. Runs bandit-adaptive scheduler (100 queries with shock at Q50)
6. Analyzes all results and generates visualizations

**Time estimate:** 2-4 hours total (depending on hardware)

### Option B: Run Individual Experiments

```bash
# Deploy specific configuration
kubectl apply -f k8s/baseline/03-manual-optimal.yaml

# Wait for pods
kubectl wait --for=condition=Ready pods --all --timeout=300s

# Get URL
RAG_URL=$(minikube service rag-app-service --url)

# Run experiment
cd benchmarks
python run_experiments.py \
  --config baseline-optimal \
  --queries 100 \
  --url $RAG_URL \
  --output ../results

# Cleanup
kubectl delete deployment --all
```

---

## Phase 4: Analyze Results

### Step 1: Run Analysis

```bash
cd benchmarks
python analyze_results.py --input ../results --output ../results/analysis
```

### Step 2: View Generated Files

```bash
ls -la ../results/analysis/

# Expected files:
# - latency_comparison.png      (bar chart)
# - percentile_comparison.png   (P50/P95/P99)
# - latency_over_time.png       (line plot showing shock)
# - cumulative_regret.png       (regret comparison)
# - comparison_table.csv        (summary table)
# - statistical_tests.json      (t-test results)
```

### Step 3: Copy Results for Dissertation

```bash
# Copy figures to your dissertation folder
cp ../results/analysis/*.png /path/to/dissertation/figures/
```

---

## Troubleshooting

### Problem: Minikube won't start

```bash
# Reset everything
minikube delete --all
docker system prune -a

# Try with less resources
minikube start --nodes=2 --memory=6144 --cpus=2
```

### Problem: GPU not detected

```bash
# Check NVIDIA driver in WSL2
nvidia-smi

# If not working, the project still works - just slower
# Generation will take ~10s on CPU vs ~0.5s on GPU
```

### Problem: Pods stuck in Pending

```bash
# Check why
kubectl describe pod <pod-name>

# Common issues:
# - Insufficient memory: Reduce Minikube memory, reduce pod requests
# - GPU unavailable: Remove nvidia.com/gpu from resource limits
```

### Problem: Model download fails

```bash
# Manually pull the model
kubectl exec -it <generation-pod-name> -- ollama pull gemma:2b

# Or use smaller model
kubectl exec -it <generation-pod-name> -- ollama pull tinyllama
```

### Problem: Connection refused to generation-service

```bash
# Check service exists
kubectl get svc generation-service

# Check pod is running
kubectl get pods -l app=generation

# Check logs
kubectl logs -l app=generation
```

### Problem: Static scheduler not scheduling pods

```bash
# Check scheduler is running
kubectl get pods -l app=static-scheduler
kubectl logs -l app=static-scheduler

# Check RBAC permissions
kubectl auth can-i create bindings --as=system:serviceaccount:default:static-scheduler

# If permission denied, reapply RBAC
kubectl apply -f k8s/static-scheduler/rbac.yaml
```

---

## Quick Reference Commands

```bash
# Start cluster
./scripts/setup_cluster.sh

# Build images
./scripts/build_all.sh

# Run all experiments
./scripts/run_all_experiments.sh

# View pods
kubectl get pods -o wide

# View pod logs
kubectl logs -f <pod-name>

# Get RAG URL
minikube service rag-app-service --url

# Delete everything
kubectl delete deployment --all
kubectl delete service --all

# Stop Minikube
minikube stop

# Destroy cluster
minikube delete
```

---

## Expected Results Summary

After running all experiments, you should see:

| Scheduler | Mean Latency | Improvement |
|-----------|--------------|-------------|
| baseline-uninformed | ~5000-10000ms | - |
| baseline-informed | ~1000-2000ms | ~60-80% |
| baseline-optimal | ~600-1000ms | ~80-90% |
| static-ml | ~600-1000ms | ~80-90% |
| bandit-adaptive | ~700-1200ms | ~75-85% |

**Key findings to discuss:**
1. Static ML approaches manual optimal placement
2. Bandit recovers after shock (visible in latency_over_time.png)
3. Statistical significance confirmed by t-tests

---

## Next Steps

1. **Copy figures** to dissertation Chapter 4
2. **Use statistics** from statistical_tests.json for significance claims
3. **Customize** the shock scenario by editing SHOCK_QUERY in bandit deployment
4. **Extend** with additional experiments if time permits
