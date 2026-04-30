# L7 Smart Gateway — FYP Submission

An L7 HTTP gateway that adaptively routes RAG/LLM inference requests across a
heterogeneous 2-node k3s cluster (one GPU node, one CPU node). Six routing
policies are compared, including a Thompson Sampling bandit with regime
detection and a 3-state circuit breaker.

This repo contains the gateway, the experiment harness, the offline ML training
pipeline for the static-classifier baseline, the k8s manifests, the test suite,
and the final aggregated results that back the dissertation.

## Cluster setup

- **GPU node**: laptop with RTX 3070, runs `ollama-gpu` (gemma:2b on GPU)
- **CPU node**: ProDesk i5-9500, runs `ollama-cpu` (gemma:2b on CPU) and the gateway pod
- **k3s** cross-host cluster (server on CPU node, agent on GPU node)
- **Model**: `gemma:2b` via Ollama on both nodes

Nodes are labelled `node_type=gpu_accelerated` / `node_type=cpu_standard`. The
gateway is pinned to the CPU node and reaches Ollama backends by service name.

## Routing modes

The `ROUTING_MODE` env var on the gateway deployment selects one of:

| Mode | Description |
|---|---|
| `baseline` | Always GPU |
| `least_in_flight` | Pick node with fewer in-flight requests |
| `static` | Random Forest classifier, 3 features (prompt length, GPU/CPU inflight) |
| `bandit_plain` | Thompson Sampling only |
| `bandit_regime` | TS + regime detector + posterior soft-reset |
| `adaptive` | TS + regime detector + 3-state circuit breaker |

All bandit logic lives in `smart-gateway/algorithms/bandit.py`. The static
model is trained offline in `ml-training/` and shipped into the gateway image
via `scripts/build_all.sh` (and tracked at `smart-gateway/model/gateway_model.pkl`
so a clean checkout still runs static mode).

## Quick start

Assuming kubectl is pointed at a 2-node k3s cluster with one GPU node:

```bash
./scripts/setup_cluster.sh                     # label nodes, install nvidia device plugin
GPU_NODE=<gpu-node-ip> ./scripts/build_all.sh  # build images + import to both nodes
./run_all.sh                                   # one full sweep over all 6 modes
./run_multiple.sh                              # 10-run suite (default config used in dissertation)
```

`run_all.sh` deploys manifests, rotates `ROUTING_MODE`, restarts pods between
modes for clean state, warms the model, then runs `benchmarks/run_experiments.py`.
`run_multiple.sh` wraps that to produce N independent runs and an aggregated
analysis.

Default per-mode workload (matches the submission): 400 queries, Poisson λ=1.0
ramping to λ=12 at the midpoint, concurrency 32, 15 s SLA timeout.

## Repo tree

```
.
├── smart-gateway/              # FastAPI L7 gateway (the system under study)
│   ├── main.py                 #   /query, /metrics, /ready, /health, /bandit/stats
│   ├── algorithms/
│   │   ├── bandit.py           #   Thompson Sampling + regime detection + 3-state CB
│   │   └── static_ml.py        #   Random Forest classifier wrapper
│   ├── vectordb.py             #   in-process ChromaDB (20-doc knowledge base)
│   ├── model/gateway_model.pkl #   tracked static model (so clean checkout still runs)
│   ├── Dockerfile              #   builds smart-gateway:v1
│   └── requirements.txt
│
├── generation-service/         # thin Ollama wrapper (one image, two deployments)
│   └── Dockerfile              #   builds generation-service:v1 — image used by ollama-gpu/cpu
│
├── k8s/                        # all manifests applied by run_all.sh
│   ├── 00-nvidia-runtime-class.yaml
│   ├── 01-ollama-gpu.yaml      #   ollama-gpu deployment + service (nodeSelector=gpu_accelerated)
│   ├── 02-ollama-cpu.yaml      #   ollama-cpu deployment + service (nodeSelector=cpu_standard)
│   └── 03-gateway-app.yaml     #   smart-gateway deployment + NodePort 32367
│
├── benchmarks/                 # experiment harness + analysis
│   ├── run_experiments.py      #   load generator: Poisson arrivals + rate ramp, port-fwd manager
│   ├── analyze_results.py      #   single-run charts + stats
│   ├── analyze_multiple.py     #   10-run aggregation: MWU, Cohen's d, Bonferroni, all figures
│   ├── prompts.json            #   prompt pool fed to the load generator
│   └── requirements.txt
│
├── ml-training/                # offline training of the static classifier
│   ├── generate_dataset.py     #   synthesises labelled (prompt, inflight) → node samples
│   ├── train_model.py          #   trains the Random Forest + writes confusion matrix etc.
│   ├── gateway_model.pkl       #   (gitignored — regenerate locally)
│   └── requirements.txt
│
├── scripts/
│   ├── setup_cluster.sh        #   node labels, NVIDIA device plugin, runtime class
│   └── build_all.sh            #   build images, copy static model, import on both nodes
│
├── tests/                      # pytest suite
│   ├── test_integration.py     #   FastAPI integration tests, all 6 routing modes
│   ├── test_bandit_logic.py    #   TS convergence + CB state machine + regime detection
│   ├── test_static_ml.py       #   loads the real .pkl
│   └── test_analyze_results.py #   smoke tests for cohens_d, MWU, phase split
│
├── docs/
│   └── TROUBLESHOOTING.md      #   common bare-metal k3s gotchas (DiskPressure, ErrImageNeverPull, etc.)
│
├── results_final/              # final figures + tables that back the dissertation
│   ├── RESULTS.md              #   headline summary (10 runs × 6 modes × 400 queries)
│   ├── aggregated_comparison_table.csv
│   ├── aggregated_statistical_tests.json
│   └── *.png                   #   latency, percentiles, success rate, recovery, etc.
│
├── results/                    # raw per-run outputs (gitignored — regenerated by run_*.sh)
│
├── run_all.sh                  # one full sweep across all 6 modes
└── run_multiple.sh             # repeat run_all.sh N times + aggregate
```

## Test suite

```bash
pytest tests/
```

Covers: bandit convergence and CB transitions, static ML against the real `.pkl`,
analyze_results stats helpers, and FastAPI integration across all 6 modes.

## Things worth knowing

- **Single replica gateway.** `_query_counter` and `inflight` are plain Python
  state guarded by asyncio locks where needed. CPython GIL + uvicorn `workers=1`
  make this safe; manifest pins `replicas: 1`.
- **Static-mode model is tracked.** Normally regenerated by `ml-training/`, but
  `smart-gateway/model/gateway_model.pkl` is committed so a clean checkout still
  runs `static` mode without retraining.
- **Rate ramp is the overload methodology.** Each run starts at λ=1 then ramps to
  λ=12 at the midpoint of the query stream, producing a normal phase and an
  overload phase that get analysed separately.
- **k3s images are per-node.** After rebuilding, you must re-import to both
  nodes — `scripts/build_all.sh` does this over SSH.
- **Bonferroni over 15 pairs.** With 6 modes, the familywise correction spans
  C(6,2)=15 comparisons; corrected p-values live in
  `results_final/aggregated_statistical_tests.json`.
- **Headline result.** Adaptive mode reduces overload-phase mean latency by 17.4%
  vs `bandit_plain` (10.9 s vs 13.1 s, Cohen's d = 0.604, p < 0.001). `bandit_plain`
  reaches GPU-baseline performance with no prior knowledge (p = 0.098, negligible
  effect — interpret as absence of evidence, not equivalence; n=10 is
  underpowered for the observed effect size).

## See also

- `results_final/RESULTS.md` — the full headline table, statistical tests, design
  rationale, known behaviours, and threats to validity
- `docs/TROUBLESHOOTING.md` — bare-metal k3s gotchas you will hit at least once
