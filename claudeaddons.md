# L7 Smart Gateway — Full Project Reference

## What This Is

FYP project. Adaptive request-level routing for RAG/LLM workloads on heterogeneous k3s cluster.

Two bare-metal nodes: laptop (RTX 3070 GPU) + ProDesk i5-9500 (CPU only). Both run Ollama serving gemma:2b. A FastAPI gateway sits in front, intercepts every /query request, does vector retrieval (ChromaDB), picks which Ollama node to forward to, proxies the generation, tracks latency, updates its routing model inline.

Four routing modes tested: baseline (always GPU), static (Random Forest classifier), bandit (Thompson Sampling), adaptive (Thompson Sampling + circuit-breaker). Evaluation: 100-query experiments with Poisson arrivals. GPU shock injected at query 50 (CPU stress + queue flood on GPU node for 90s). 10 runs aggregated.

Key finding: adaptive mode is best when circuit-breaker fires correctly (2-9/50 post-shock failures), but bimodal — sometimes fails completely (33-50/50) when regime detection misses. See "Known Issues" section.

---

## Architecture

```
                    ┌──────────────────────┐
                    │   Benchmark Client   │
                    │  (run_experiments.py) │
                    └──────────┬───────────┘
                               │ HTTP POST /query
                               ▼
                    ┌──────────────────────┐
                    │    Smart Gateway     │  ← 1 replica, CPU node
                    │      (FastAPI)       │
                    │                      │
                    │  1. Vector retrieval  │  ChromaDB in-memory
                    │  2. Route decision   │  baseline|static|bandit|adaptive
                    │  3. Proxy generation │
                    │  4. Update model     │  bandit posteriors / reward
                    └────────┬────┬────────┘
                             │    │
                    ┌────────┘    └────────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌─────────────────┐
          │  Ollama (GPU)   │    │  Ollama (CPU)    │
          │  arm 0          │    │  arm 1           │
          │  laptop node    │    │  ProDesk node    │
          │  RTX 3070       │    │  i5-9500         │
          └─────────────────┘    └─────────────────┘
```

Gateway = single replica. In-memory state: inflight counters, bandit posteriors, reward windows. Not shared. Must be 1 process.

---

## Directory Tree

```
fyp/
├── smart-gateway/                 # THE APPLICATION
│   ├── main.py                    # FastAPI app. /query endpoint. 3-stage pipeline:
│   │                              #   retrieval → routing → generation proxy.
│   │                              #   Tracks inflight counts per node under asyncio lock.
│   │                              #   Updates bandit after every request (including failures).
│   ├── vectordb.py                # ChromaDB wrapper. 20 hardcoded docs. Cosine similarity.
│   │                              #   all-MiniLM-L6-v2 embeddings. CPU-bound, run in executor.
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── bandit.py              # Thompson Sampling (Beta-Bernoulli). Two arms (GPU/CPU).
│   │   │                          #   Reward: exp(-k * excess / target). Regime detection
│   │   │                          #   via sliding window. Soft-reset on regime change.
│   │   │                          #   Adaptive mode: circuit-breaker blocks worse arm for 90s.
│   │   └── static_ml.py           # Random Forest wrapper. 3 features: prompt_length,
│   │                              #   gpu_in_flight, cpu_in_flight. Loads gateway_model.pkl.
│   ├── Dockerfile                 # python:3.11-slim. Pre-downloads embedding models.
│   │                              #   Single uvicorn worker. HEALTHCHECK on /health.
│   ├── requirements.txt           # fastapi, httpx, chromadb, sentence-transformers,
│   │                              #   scikit-learn, numpy, pydantic
│   └── model/                     # CREATED AT BUILD TIME by build_all.sh
│       └── gateway_model.pkl      #   Copied from ml-training/. Gitignored.
│
├── generation-service/            # THIN OLLAMA WRAPPER
│   ├── Dockerfile                 # FROM ollama/ollama:latest. Model pulled at runtime
│   │                              #   via k8s postStart lifecycle hook (ollama pull gemma:2b).
│   └── pull-model.sh              # Helper script for manual model pull. Not used in k8s.
│
├── k8s/                           # KUBERNETES MANIFESTS (applied in numeric order)
│   ├── 00-nvidia-runtime-class.yaml  # RuntimeClass "nvidia". Must exist before GPU pods.
│   ├── 01-ollama-gpu.yaml         # Deployment + Service. nodeSelector: gpu_accelerated.
│   │                              #   runtimeClassName: nvidia. Mounts /dev/nvidia* devices.
│   │                              #   privileged: true. Requests nvidia.com/gpu: 1.
│   │                              #   OLLAMA_NUM_PARALLEL=2, OLLAMA_MAX_QUEUE=50.
│   ├── 02-ollama-cpu.yaml         # Deployment + Service. nodeSelector: cpu_standard.
│   │                              #   No GPU resources. Same parallel/queue config.
│   ├── 03-gateway-app.yaml        # Deployment + Service (NodePort). nodeSelector: cpu_standard.
│   │                              #   replicas: 1. Env vars: ROUTING_MODE, TARGET_LATENCY_MS,
│   │                              #   GENERATION_TIMEOUT, BANDIT_WINDOW_SIZE, etc.
│   │                              #   run_all.sh overrides ROUTING_MODE via kubectl set env.
│   └── 04-shock-job.yaml          # Job. nodeSelector: gpu_accelerated. Two containers:
│                                  #   1) polinux/stress --cpu 4 --timeout 90s
│                                  #   2) curlimages/curl flood loop (8 concurrent, 90s)
│                                  #   ttlSecondsAfterFinished: 60.
│
├── ml-training/                   # OFFLINE MODEL TRAINING (for static mode)
│   ├── generate_dataset.py        # Synthetic data. 3000 samples. Expert routing rules:
│   │                              #   gpu_in_flight>8 → CPU, >5+long_prompt → CPU, else GPU.
│   │                              #   5% label noise. Output: training_data.csv
│   ├── train_model.py             # RandomForest (200 trees, max_depth=10, balanced).
│   │                              #   5-fold CV. Output: gateway_model.pkl + report + charts.
│   └── requirements.txt           # scikit-learn, pandas, numpy, matplotlib, seaborn
│
├── benchmarks/                    # EXPERIMENT RUNNER + ANALYSIS
│   ├── run_experiments.py         # Async experiment runner. Poisson arrivals (default 2 req/s).
│   │                              #   Semaphore-bounded concurrency (default 4). kubectl applies
│   │                              #   shock job at dispatch index 50. PortForwardManager for
│   │                              #   WSL2 reliability. Retry logic on ConnectError. Outputs
│   │                              #   {config}_results.json and {config}_summary.json.
│   ├── analyze_results.py         # Single-run analysis. 8 charts + statistical tests.
│   │                              #   Mann-Whitney U (primary), Welch t-test, KS test.
│   │                              #   Bonferroni correction (6 pairwise comparisons).
│   │                              #   Bootstrap 95% CI. Cohen's d effect size.
│   ├── analyze_multiple.py        # Multi-run aggregation. Loads run_1..run_N results.
│   │                              #   Boxplots, averaged time series, aggregated stats.
│   ├── inject_shock.sh            # STALE. References old k8s/shock/ path. Not used.
│   │                              #   run_experiments.py applies shock directly.
│   ├── prompts.json               # 20 diverse prompts covering k8s, ML, RAG, scheduling.
│   └── requirements.txt           # httpx, pandas, numpy, matplotlib, seaborn, scipy, tqdm
│
├── scripts/                       # SETUP + BUILD
│   ├── setup_cluster.sh           # One-time cluster config. Labels nodes, applies RuntimeClass,
│   │                              #   installs NVIDIA device plugin, patches runtimeClassName,
│   │                              #   waits for GPU resource to register.
│   └── build_all.sh               # Builds Docker images. Copies gateway_model.pkl into
│                                  #   smart-gateway/model/. docker save | k3s ctr images import
│                                  #   on both nodes (local + SSH to GPU node).
│
├── run_all.sh                     # TOP-LEVEL ORCHESTRATOR. Runs 4 experiments sequentially.
│                                  #   switch_mode: kubectl set env + rollout restart all pods.
│                                  #   Waits for pods + model pull + GPU VRAM warmup.
│                                  #   Calls run_experiments.py per mode, then analyze_results.py.
│
├── run_multiple.sh                # MULTI-RUN WRAPPER. Loops run_all.sh N times.
│                                  #   Currently fragile — needs robustness improvements.
│                                  #   See "Planned Improvements" section.
│
├── docs/
│   └── TROUBLESHOOTING.md         # STALE. References old k3d/scheduler architecture.
│
├── results/                       # GENERATED. 10 runs + aggregated analysis. Gitignored.
├── archive/                       # Old architecture code (k8s scheduler, rag-app). Gitignored.
├── addons.txt                     # Personal notes + planned improvements.
└── .gitignore
```

---

## Component Details

### smart-gateway/main.py — Request Pipeline

Every `/query` request:

1. **Retrieval** — `vectordb.retrieve(prompt, top_k=3)`. ChromaDB cosine similarity search. Returns top-3 matching documents from 20-doc knowledge base. Runs in thread executor (CPU-bound embedding).

2. **Augment** — Prepends retrieved context to user prompt. Format: `"Context:\n- doc1\n- doc2\n\nQuestion: {prompt}\n\nAnswer:"`. If `include_context=False`, raw prompt passed through.

3. **Route** — Under `_inflight_lock`:
   - `baseline`: return 0 (GPU always)
   - `static`: `static_router.predict(prompt, gpu_inflight, cpu_inflight)` — RF model
   - `bandit`: `bandit.select_arm()` — Thompson Sampling posterior draw
   - `adaptive`: same as bandit but circuit-breaker may exclude blocked arm

4. **Increment** inflight counter for chosen node. Release lock.

5. **Generate** — `httpx.AsyncClient.post(f"{node_url}/api/generate", json=payload)`. Non-streaming. Options: temperature=0.7, num_predict=256, top_p=0.9. Timeout: GENERATION_TIMEOUT (30s).

6. **Error handling** — Catches TimeoutException (504), ConnectError (503), RemoteProtocolError (502), ReadError (502), HTTPStatusError (502). All paths decrement inflight counter in `finally`.

7. **Update** — If bandit/adaptive mode: `bandit.update(node, generation_time_ms, timed_out=request_failed)`. Failures penalized with reward=0.

8. **Record** — Append to `metrics_history` deque (capped at 500). Return QueryResponse.

### algorithms/bandit.py — Thompson Sampling + Circuit-Breaker

**Posterior**: Beta(alpha, beta) per arm. Initial: Beta(1,1) = uniform.

**Reward function**: `exp(-k * max(0, latency - target) / target)`
- latency == target → reward = 1.0
- latency = 2x target → reward = exp(-1) ≈ 0.37
- timeout → reward = 0.0

**Selection**: Sample from each arm's posterior. Pick highest. In adaptive mode, blocked arms excluded (safety fallback: if all blocked, include all).

**Update**: `alpha[arm] += reward`, `beta[arm] += (1 - reward)`. Append to sliding window.

**Regime detection**: After burn-in (window_size * 3 = 15 pulls):
- Track `baseline_reward` (slowly drifting EMA of window average)
- If `current_window_avg < baseline * 0.4` → soft reset

**Soft reset**: Decay posteriors: `0.3 * old + 0.7 * prior`. Clear window. Reset baseline.

**Circuit-breaker** (adaptive only): On soft reset, identify worse arm by per-arm reward window average. Block it for `recovery_window_s` (90s). Only blocks if other arm is currently unblocked (prevents deadlock on dual failure).

### algorithms/static_ml.py — Random Forest Router

Loads `gateway_model.pkl`. Extracts 3 features: word count of augmented prompt, GPU inflight, CPU inflight. Returns 0 (GPU) or 1 (CPU). No online learning — frozen at training-time knowledge.

### vectordb.py — RAG Retrieval

20 hardcoded documents covering: Kubernetes (5), ML (5), RAG/LLM (5), scheduling (5). ChromaDB ephemeral client (in-memory, no persistence). Default embedding: all-MiniLM-L6-v2 (downloaded at Docker build time). Cosine similarity.

### Shock Design (k8s/04-shock-job.yaml)

Two-container Job on GPU node:
1. **CPU stress**: `polinux/stress --cpu 4 --timeout 90s`. Saturates CPU cores. Slows Ollama tokenization/preprocessing.
2. **Queue flood**: curl loop firing long-prompt generation requests at ollama-gpu-service. 8 concurrent (capped to avoid cluster-wide network blackout). 90s duration.

Combined effect: GPU Ollama becomes slow + queue-saturated. Bandit/adaptive should detect latency spike and route to CPU.

---

## Data Flow

```
setup_cluster.sh     →  one-time: label nodes, install NVIDIA plugin
                     ↓
build_all.sh         →  docker build smart-gateway + generation-service
                     →  copy gateway_model.pkl into smart-gateway/model/
                     →  docker save | k3s ctr images import (both nodes)
                     ↓
kubectl apply k8s/   →  deploy ollama-gpu, ollama-cpu, smart-gateway
                     ↓
run_all.sh           →  for each mode (baseline, static, bandit, adaptive):
                     →    kubectl set env ROUTING_MODE=X
                     →    rollout restart all pods
                     →    wait for pods + model pull + VRAM warmup
                     →    run_experiments.py --config X
                     →      dispatches 100 queries (Poisson, rate=0.9)
                     →      applies shock job at query 50
                     →      writes {mode}_results.json + {mode}_summary.json
                     →    cleanup shock job
                     →  analyze_results.py → charts + statistical_tests.json
                     ↓
run_multiple.sh      →  loops run_all.sh 10 times → results/run_{1..10}/
                     →  analyze_multiple.py → aggregated analysis
```

---

## Known Issues

### 1. Adaptive Mode Bimodal Failure

**Symptom**: Adaptive post-shock results are either excellent (2-9 failures) or catastrophic (33-50 failures). No middle ground.

**Data from 10 runs** (post-shock failures out of 50):

| Run | Baseline | Static | Bandit | Adaptive |
|-----|----------|--------|--------|----------|
| 1   | 49       | 50     | 16     | **50**   |
| 2   | 50       | 44     | 7      | **50**   |
| 3   | 49       | 49     | 20     | **33**   |
| 4   | 48       | 48     | 29     | **5**    |
| 5   | 49       | 50     | 16     | **5**    |
| 6   | 50       | 47     | 11     | **48**   |
| 7   | 49       | 48     | 27     | **33**   |
| 8   | 50       | 49     | 25     | **2**    |
| 9   | 50       | 48     | 26     | **9**    |
| 10  | 49       | 48     | 33     | **48**   |

**Root cause** — Two compounding problems:

**A. Regime detection is stochastic.** Window size = 5. If Thompson Sampling happens to route several queries to CPU right as shock hits (random posterior draws), the GPU reward window doesn't fill with bad observations. Regime change never fires. Adaptive then behaves like vanilla bandit without the benefit of soft-reset — posteriors are stale-strong for GPU.

**B. Recovery window == shock duration.** Both are 90s. Detection delay is ~5-15 queries after shock starts (~5-17s at 0.9 req/s). Block starts at shock+5s to shock+17s. Block expires at shock+85s to shock+107s. Narrow overlap where block expires while shock still active. In bad cases, traffic resumes to still-stressed GPU.

**C. 503 cascade.** In the worst runs (1,2,6,10), failures are HTTP 503 (ollama unreachable), not timeouts. The shock kills the GPU Ollama process entirely. If circuit-breaker hasn't fired, all traffic hits dead GPU → 503. Even if it fires, CPU Ollama on ProDesk may be resource-constrained from hours of sequential experiments.

**Suggested fixes** (pick any combination):

1. **Increase recovery window to 120s** (> shock 90s + detection lag). Simple, no downside for experiments.
2. **Reduce window_size to 3**. Faster detection. Risk: more false positives in normal operation. Acceptable for 100-query experiments.
3. **Force GPU sampling during detection window**. After burn-in, ensure at least 1 of every 3 queries goes to GPU regardless of posterior. Guarantees regime detector has fresh GPU data. Adds slight exploration cost.
4. **Immediate block on 503**. If a request returns 503 (unreachable), block that arm for recovery_window_s immediately without waiting for regime detection. 503 = node is dead, no need to accumulate evidence.

Fix #4 is highest-value, lowest-risk. A 503 is unambiguous signal — no statistical detection needed.

### 2. run_multiple.sh Fragility

- Hardcoded `PROJECT_ROOT="/home/ike/vsc/fyp"`
- `set -e` kills entire suite on first failure. No retry.
- No DiskPressure preflight check (ProDesk has disk issues)
- No temp-dir isolation. Failed run leaves partial results in final directory.
- Planned fixes in addons.txt are all correct. See items 3-8.

### 3. inject_shock.sh Stale Path

References `k8s/shock/cpu-stress.yaml`. Doesn't exist. Not used by anything — `run_experiments.py` applies `k8s/04-shock-job.yaml` directly. Safe to delete or fix.

### 4. TROUBLESHOOTING.md Stale

References k3d, old scheduler pods, old RBAC, old service names. Needs rewrite for k3s bare-metal + L7 gateway.

### 5. .gitignore Incomplete

Currently only ignores `.claude/` and `/archive/`. Missing: `results/`, `__pycache__/`, `smart-gateway/model/`, `*.pyc`, `addons.txt`, `.env`.

---

## Replication Guide (from zero)

### Hardware

- GPU node: any Linux machine with NVIDIA GPU. k3s agent.
- CPU node: any Linux machine. k3s server. Needs ~15GB disk free for Ollama model.
- Both on same LAN. SSH access from CPU→GPU node.

### Software Prerequisites

- k3s installed on both nodes (server on CPU, agent on GPU)
- Docker on CPU node (for building images)
- NVIDIA container toolkit on GPU node
- Python 3.11+ on CPU node (for benchmarks)
- pip packages: see benchmarks/requirements.txt + ml-training/requirements.txt

### Steps

```bash
# 1. Clone
git clone <repo> && cd fyp

# 2. Setup cluster (one-time)
./scripts/setup_cluster.sh

# 3. Train static model (optional, only for static mode)
cd ml-training
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
cd ..

# 4. Build + import images
GPU_NODE=<gpu-ip> ./scripts/build_all.sh

# 5. Deploy
kubectl apply -f k8s/

# 6. Wait for model pulls (~2-5 min)
kubectl get pods -w

# 7. Install benchmark deps
pip install -r benchmarks/requirements.txt

# 8. Single experiment run
./run_all.sh

# 9. Multi-run (10x)
./run_multiple.sh
```

### Environment Variables (run_all.sh)

| Variable | Default | Purpose |
|----------|---------|---------|
| RESULTS_DIR | results/YYYYMMDD_HHMMSS | Output directory |
| QUERIES | 100 | Queries per experiment |
| RATE | 0.9 | Mean arrival rate (req/s) |
| CONCURRENCY | 2 | Max in-flight requests |

### Gateway Environment Variables (k8s/03-gateway-app.yaml)

| Variable | Default | Purpose |
|----------|---------|---------|
| ROUTING_MODE | bandit | baseline\|static\|bandit\|adaptive |
| TARGET_LATENCY_MS | 5000 | Reward function target |
| GENERATION_TIMEOUT | 30 | Seconds before 504 |
| BANDIT_WINDOW_SIZE | 5 | Regime detection window |
| MODEL_NAME | gemma:2b | Ollama model |
| STATIC_MODEL_PATH | /app/model/gateway_model.pkl | RF model location |

---

## Refactoring Suggestions

### Priority 1 — Fix adaptive reliability

Implement fix #4 from Known Issues (immediate arm block on 503). Optionally increase recovery window to 120s. This directly improves thesis results.

### Priority 2 — Harden run_multiple.sh

Implement addons.txt items 3-8. The current script will silently produce partial/corrupt multi-run results if any experiment fails.

### Priority 3 — .gitignore

Add: `results/`, `__pycache__/`, `smart-gateway/model/`, `*.pyc`, `addons.txt`, `.env`, `claudeaddons.md`.

### Priority 4 — Delete dead files

- `benchmarks/inject_shock.sh` — stale, not used
- `generation-service/pull-model.sh` — not used (k8s postStart hook does the pull inline)

### Priority 5 — Update TROUBLESHOOTING.md

Rewrite for current architecture before thesis submission.

### Would-do-differently-from-scratch (not worth changing now)

- **Gateway as single Go binary** instead of Python. Eliminates GIL, simpler deployment, no sentence-transformers download at build time. But Python was right for prototyping speed.
- **Separate retrieval service** from routing. Current design couples ChromaDB into the gateway pod. Fine for FYP scale. At scale, retrieval = separate service, gateway = pure router.
- **Redis or etcd for inflight state** instead of in-memory. Removes single-replica constraint. Overkill for 2-node FYP.
- **Prometheus metrics** instead of in-memory deque. Native k8s observability. But the deque works fine for 100-query experiments.
- **Helm chart or Kustomize** instead of flat YAML + kubectl set env. More reproducible mode switching. But shell scripts are easier to debug for a solo project.

---

## Session Summary — 2026-04-12/13 (Phase 2 + Production Results)

### What changed from the state above

**This entire document above is now partially outdated.** The project was upgraded from smart-gateway:v4 (4 modes, shock injection) to smart-gateway:v5 (6 modes, rate ramp, 3-state CB). Key changes:

#### Gateway v5 — 6 routing modes (algorithms/bandit.py rewritten)

| Mode | Description |
|------|-------------|
| `baseline` | Always GPU (unchanged) |
| `least_in_flight` | Route to node with fewer in-flight requests; GPU wins ties (new) |
| `static` | Random Forest classifier (unchanged) |
| `bandit_plain` | Thompson Sampling only — no regime detection, no CB (new) |
| `bandit_regime` | Thompson Sampling + regime detection + posterior soft-reset (was `bandit`) |
| `adaptive` | bandit_regime + **3-state circuit breaker** CLOSED→OPEN→HALF_OPEN (replaces old timer CB) |

**Old `bandit` mode = new `bandit_regime`. Old `adaptive` timer-block = replaced by proper 3-state CB.**

#### 3-state circuit breaker (algorithms/bandit.py)

- **CLOSED**: normal operation
- **OPEN**: arm blocked for `cb_open_duration_s=15s` after regime change detected
- **HALF_OPEN**: graduated probe admission — 25% for first 15s, then 50% for next 15s; mean probe reward ≥0.3 → CLOSED; 3+ probes <0.1 → back to OPEN

Constructor flags: `enable_regime_detection`, `enable_cb` (separates the three variants cleanly).

#### Shock → Rate ramp

Old: artificial `04-shock-job.yaml` (CPU stress + GPU queue flood at query 50).
New: Poisson rate ramp from λ=1.0 → λ=12.0 at `n_queries // 2` (dynamic midpoint). No external job. Natural overload from traffic alone.

#### run_all.sh — key additions
- `GENERATION_TIMEOUT` env var passed to gateway via `kubectl set env` (SLA lever)
- `warm_cpu_ram()` — pre-warms CPU node before each mode (eliminates cold-start penalty)
- `QUERIES=400, RAMP_RATE=12.0, CONCURRENCY=32` stress preset
- Config banner logged at start of each run

#### run_multiple.sh — fixed
- Individual run dirs now named `YYYYMMDD_HHMMSS` (matching run_all.sh), not `run_1`, `run_2`
- Env vars (`GENERATION_TIMEOUT`, `QUERIES`, `RAMP_RATE`, `CONCURRENCY`) exported and passed through
- 6-mode failure detection (was hardcoded to 4 modes)

#### analyze_multiple.py — fixed
- `_discover_run_dirs()` discovers both old `run_N` and new `YYYYMMDD_HHMMSS` directories by regex
- Backward-compatible

---

### SLA Calibration Progression

| Run dir | Timeout | Queries | λ ramp | Baseline success | Finding |
|---------|---------|---------|--------|------------------|---------|
| 20260411_164639 | 30s | 100 | 8.0 | 100% | Too lenient — GPU never fails |
| 20260411_171323 | 30s | 100 | 12.0 | 100% | Same — arrival rate can't break GPU at 30s |
| 20260411_215535 | 10s | 100 | 12.0 | 91% | Too tight — CPU also times out, offload useless |
| 20260411_235854 | 15s | 100 | 12.0 | 100% | Only 50 overload queries — not enough pressure |
| 20260412_002239 | 15s | 400 | 12.0 | 91.5% | **First real differentiation** — 6 modes, static 77%, bandit_regime 86% |
| 20260412_113759 | 20s | 400 | 12.0 | 99.5% | Too lenient again — baseline near-perfect |
| multi_20260412_172649 | **15s** | **400** | **12.0** | 89.9% | **Production suite — 10/10 runs, final results** |

**Chosen config: Q=400, rate=1.0→12.0, concurrency=32, GENERATION_TIMEOUT=15s.**

---

### Final Results (10-run suite, 4000 queries per mode)

| Mode | Success | Mean (ms) | Overload mean (ms) | % GPU |
|------|---------|-----------|--------------------|-------|
| Baseline | 89.9% | 6,711 | 13,123 | 100% |
| Bandit (plain) | 89.2% | 6,828 | 13,160 | 99% |
| Bandit (regime) | 87.5% | 5,918 | 11,506 | 99% |
| Adaptive | 85.7% | 5,567 | 10,872 | 99% |
| Static ML | 80.7% | 3,545 | 6,083 | 97% |
| Least In-Flight | 80.3% | 4,754 | 8,362 | 96% |

**Key stats:**
- `baseline vs bandit_plain`: p=0.098 (non-sig), d=−0.019 → TS matches oracle with zero config
- `bandit_plain vs adaptive (overload)`: p<0.001, d=0.604 (medium) → adaptive 17.4% lower overload latency
- `bandit_regime vs adaptive (full)`: p=0.151 (non-sig), d=0.063 → CB adds negligible improvement over regime detection alone at this SLA

**Why CPU offload doesn't improve success rate:** At λ=12, the CPU node's queue overflows under the surge (CPU throughput ≈1 query/15s). Routing even 2–4% to CPU during overload produces more timeouts than it saves. Bandit_plain correctly learns to stay on GPU. Static/LIF blindly route to CPU and pay the penalty.

**Thesis story:** bandit_plain is the success-rate protagonist (≈ baseline, no prior knowledge, beats trained static by 8.5pp). Adaptive is the latency protagonist (best overload latency, medium effect vs bandit_plain). Static/LIF are negative results.

---

### Files updated in this session

| File | Change |
|------|--------|
| `smart-gateway/algorithms/bandit.py` | Rewritten: CBState enum, enable_regime_detection + enable_cb flags, 3-state CB |
| `smart-gateway/main.py` | 6 modes, new env vars (CB_OPEN_DURATION_S, CB_HALF_OPEN_DURATION_S) |
| `benchmarks/run_experiments.py` | Dynamic midpoint ramp, overload_query in summary JSON |
| `benchmarks/analyze_results.py` | 6-mode EXPERIMENT_ORDER/COLORS/LABELS, dynamic pairs, overload_query from JSON |
| `benchmarks/analyze_multiple.py` | Same 6-mode updates + `_discover_run_dirs()` for timestamped dirs |
| `run_all.sh` | GENERATION_TIMEOUT, warm_cpu_ram(), 6 modes, config banner |
| `run_multiple.sh` | Timestamped dirs, env var passthrough, 6-mode failure detection |
| `k8s/03-gateway-app.yaml` | Image bumped to smart-gateway:v5 |
| `FYP_Draft_Updated(1).md` | Methodology (§3), implementation (§4), results (§5), conclusion (§6) rewritten for v5 |
| `results/*/config.txt` | Created for each run dir documenting parameters and observations |
