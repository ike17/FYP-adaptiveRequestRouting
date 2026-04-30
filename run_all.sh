#!/bin/bash
# One full sweep across all 6 routing modes: deploy, warm, run, analyse.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/$(date +%Y%m%d_%H%M%S)}"
QUERIES="${QUERIES:-100}"
RATE="${RATE:-1.0}"
RAMP_RATE="${RAMP_RATE:-4.0}"
CONCURRENCY="${CONCURRENCY:-16}"
GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-30}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:32367}"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

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

wait_for_gateway_ready() {
    local max=120 waited=0
    log "checking gateway readiness at $GATEWAY_URL/ready..."
    while [ $waited -lt $max ]; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "$GATEWAY_URL/ready" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            log "  gateway ready"; return 0
        elif [ "$code" = "503" ]; then
            log "  gateway reports nodes not ready (503)"
            return 1
        fi
        sleep 10; waited=$((waited + 10))
        log "  waiting for gateway... (${waited}s/${max}s, last HTTP $code)"
    done
    log "WARNING: gateway readiness timeout"
    return 2
}

handle_gateway_readiness() {
    wait_for_gateway_ready
    local rc=$?
    if [ $rc -eq 0 ]; then return 0; fi

    if [ $rc -eq 1 ]; then
        log "  attempting recovery: restarting ollama-cpu..."
        kubectl rollout restart deployment/ollama-cpu
        kubectl rollout status deployment/ollama-cpu --timeout=120s || true
        wait_for_pods "app=ollama-cpu"
        wait_for_model "app=ollama-cpu"

        wait_for_gateway_ready
        local rc2=$?
        if [ $rc2 -eq 0 ]; then
            log "  recovery successful"; return 0
        fi
        log "ERROR: gateway still not ready after ollama-cpu restart (rc=$rc2)"
        return 1
    fi

    if [ $rc -eq 2 ]; then
        log "  gateway unreachable, retrying after 10s..."
        sleep 10
        wait_for_gateway_ready
        local rc2=$?
        if [ $rc2 -eq 0 ]; then return 0; fi
        log "ERROR: gateway still unreachable after retry (rc=$rc2)"
        return 1
    fi
}

switch_mode() {
    local mode="$1"
    log "switching gateway to mode: $mode (timeout=${GENERATION_TIMEOUT}s) — restarting all pods for clean state"
    kubectl set env deployment/smart-gateway ROUTING_MODE="$mode" GENERATION_TIMEOUT="$GENERATION_TIMEOUT"
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
    warm_cpu_ram
    handle_gateway_readiness || { log "ERROR: gateway not ready, aborting $mode"; return 1; }
}

warm_gpu_vram() {
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

warm_cpu_ram() {
    local max_attempts=4 attempt=0 interval=15
    local pod
    pod=$(kubectl get pod -l "app=ollama-cpu" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    log "  pre-warming CPU RAM on pod $pod (up to $((max_attempts * interval * 2))s)..."
    while [ $attempt -lt $max_attempts ]; do
        if kubectl exec "$pod" -- ollama run gemma:2b "hello" > /dev/null 2>&1; then
            log "  CPU RAM warm (attempt $((attempt + 1)))"
            return 0
        fi
        attempt=$((attempt + 1))
        log "  CPU model still loading... ($attempt/$max_attempts)"
        sleep $interval
    done
    log "WARNING: CPU RAM pre-warm inconclusive — proceeding anyway"
}

run_experiment() {
    local mode="$1"
    log "Experiment: $mode"

    switch_mode "$mode"

    python3 "$PROJECT_ROOT/benchmarks/run_experiments.py" \
        --config "$mode" \
        --queries "$QUERIES" \
        --rate "$RATE" \
        --ramp-rate "$RAMP_RATE" \
        --concurrency "$CONCURRENCY" \
        --output "$RESULTS_DIR" \
        --url "$GATEWAY_URL" || true

    log "  experiment $mode DONE"
}

log "Results directory: $RESULTS_DIR"
log "Config: Q=$QUERIES rate=$RATE ramp=$RAMP_RATE conc=$CONCURRENCY timeout=${GENERATION_TIMEOUT}s"

log "Deploying manifests..."
kubectl apply -f "$PROJECT_ROOT/k8s/01-ollama-gpu.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/02-ollama-cpu.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/03-gateway-app.yaml"

MODES=(baseline least_in_flight static bandit_plain bandit_regime adaptive)

MODES=($(printf '%s\n' "${MODES[@]}" | shuf))
log "Experiment order: ${MODES[*]}"

for mode in "${MODES[@]}"; do
    run_experiment "$mode"
    log "  cooldown 30s between modes..."
    sleep 30
done

log "All experiments done"
python3 "$PROJECT_ROOT/benchmarks/analyze_results.py" \
    --input "$RESULTS_DIR" \
    --output "$RESULTS_DIR/analysis"

log "Complete"
log "Results: $RESULTS_DIR"
log "Charts:  $RESULTS_DIR/analysis/"
