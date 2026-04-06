#!/bin/bash
# run_all.sh — L7 Smart Gateway experiment suite
#
# Runs four experiments back-to-back (baseline, static, bandit, adaptive) against
# the same pair of Ollama pods. Between experiments the gateway's ROUTING_MODE is
# changed via `kubectl set env` + rollout restart so each experiment starts with
# a fresh pod and clean in-memory state (bandit posteriors, queue counters, etc.)
# The "adaptive" mode adds a circuit-breaker on top of Thompson Sampling: the
# worse arm is blocked for ~90s after a regime change detected during the shock.
#
# Prerequisites:
#   1. Cluster running:  ./scripts/setup_cluster.sh
#   2. Images built:     ./scripts/build_all.sh
#   3. Model trained:    cd ml-training && python generate_dataset.py && python train_model.py
#                        (only required for static mode)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/$(date +%Y%m%d_%H%M%S)}"
QUERIES="${QUERIES:-100}"
RATE="${RATE:-0.9}"
CONCURRENCY="${CONCURRENCY:-2}"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Cleanup ────────────────────────────────────────────────────────────────

cleanup_shock() {
    kubectl delete job cpu-shock --ignore-not-found 2>/dev/null || true
}

trap 'cleanup_shock' EXIT

# ─── Wait helpers ────────────────────────────────────────────────────────────

wait_for_pods() {
    local label="$1" max=300 waited=0
    log "waiting for pods matching label='$label'..."
    while [ $waited -lt $max ]; do
        local not_ready
        not_ready=$(kubectl get pods -l "$label" --no-headers 2>/dev/null \
            | awk 'NF && !/Running|Completed|Error|Terminating|UnexpectedAdmissionError/' | wc -l)
        local running
        running=$(kubectl get pods -l "$label" --no-headers 2>/dev/null \
            | awk '/Running/' | wc -l)
        if [ "$not_ready" -eq 0 ] && [ "$running" -ge 1 ]; then
            log "  pods ready ($running running)"; return 0
        fi
        log "  waiting... (${waited}s/${max}s, running=$running, not_ready=$not_ready)"
        sleep 15; waited=$((waited + 15))
    done
    log "ERROR: pod readiness timeout"; kubectl get pods; return 1
}

wait_for_model() {
    # The postStart lifecycle hook pulls the model asynchronously — the pod can
    # be "Running" before `ollama list` shows gemma:2b. Poll until it appears.
    local pod_label="$1" max=300 interval=10 waited=0
    log "waiting for gemma:2b to load on $pod_label (up to ${max}s)..."
    while [ $waited -lt $max ]; do
        local pod
        pod=$(kubectl get pod -l "$pod_label" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [ -n "$pod" ] && kubectl exec "$pod" -- ollama list 2>/dev/null | grep -q "gemma:2b"; then
            log "  model ready ($pod, ${waited}s elapsed)"; return 0
        fi
        sleep $interval; waited=$((waited + interval))
        log "  model loading... (${waited}/${max}s)"
    done
    log "WARNING: model load timeout — proceeding anyway"
    return 0
}

# ─── Per-experiment logic ────────────────────────────────────────────────────

switch_mode() {
    local mode="$1"
    log "switching gateway to mode: $mode — restarting all pods for clean state"
    kubectl set env deployment/smart-gateway ROUTING_MODE="$mode"
    kubectl rollout restart deployment/ollama-gpu deployment/ollama-cpu deployment/smart-gateway
    kubectl rollout status deployment/ollama-gpu    --timeout=120s || true
    kubectl rollout status deployment/ollama-cpu    --timeout=120s || true
    kubectl rollout status deployment/smart-gateway --timeout=120s || true
    wait_for_pods "app=ollama-gpu"
    wait_for_pods "app=ollama-cpu"
    wait_for_pods "app=smart-gateway"
    wait_for_model "app=ollama-gpu"
    wait_for_model "app=ollama-cpu"
    warm_gpu_vram
}

warm_gpu_vram() {
    # ollama list confirms the model file exists, but VRAM loading is lazy — it
    # only happens on the first inference request.  Run a short generation directly
    # in the pod (bypassing the gateway's 30s timeout) and block until it succeeds.
    # This guarantees VRAM is hot before the benchmark starts, preventing a cold-GPU
    # 504 from poisoning the bandit's initial prior.
    local max_attempts=6 attempt=0 interval=15
    local pod
    pod=$(kubectl get pod -l "app=ollama-gpu" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    log "  pre-warming GPU VRAM on pod $pod (up to $((max_attempts * interval * 2))s)..."
    while [ $attempt -lt $max_attempts ]; do
        if kubectl exec "$pod" -- ollama run gemma:2b "hello" > /dev/null 2>&1; then
            log "  GPU VRAM warm (attempt $((attempt + 1)))"
            return 0
        fi
        attempt=$((attempt + 1))
        log "  VRAM still loading... ($attempt/$max_attempts)"
        sleep $interval
    done
    log "WARNING: GPU VRAM pre-warm inconclusive — proceeding anyway"
}

run_experiment() {
    local mode="$1"
    log "=== EXPERIMENT: $mode ==="

    switch_mode "$mode"

    # PortForwardManager inside run_experiments.py owns the tunnel and auto-restarts
    # it if the shock or pod churn kills the connection mid-experiment.
    python3 "$PROJECT_ROOT/benchmarks/run_experiments.py" \
        --config "$mode" \
        --queries "$QUERIES" \
        --rate "$RATE" \
        --concurrency "$CONCURRENCY" \
        --output "$RESULTS_DIR" \
        --url "http://localhost:32367" || true

    cleanup_shock
    log "  experiment $mode DONE"
}

# ─── Main ────────────────────────────────────────────────────────────────────

log "Results directory: $RESULTS_DIR"

log "Deploying manifests..."
kubectl apply -f "$PROJECT_ROOT/k8s/01-ollama-gpu.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/02-ollama-cpu.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/03-gateway-app.yaml"

run_experiment "baseline"
run_experiment "static"
run_experiment "bandit"
run_experiment "adaptive"

log "=== ALL EXPERIMENTS DONE ==="
python3 "$PROJECT_ROOT/benchmarks/analyze_results.py" \
    --input "$RESULTS_DIR" \
    --output "$RESULTS_DIR/analysis"

log "=== COMPLETE ==="
log "Results: $RESULTS_DIR"
log "Charts:  $RESULTS_DIR/analysis/"
