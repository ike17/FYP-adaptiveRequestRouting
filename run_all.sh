#!/bin/bash
set -e

PROJECT_ROOT="/c/Users/shaik/Nextcloud/Lectures/FYP/fyp-rag-scheduler/fyp-rag-scheduler"
RESULTS_DIR="$PROJECT_ROOT/results"
QUERIES=100
DELAY=0.5

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_pods() {
    local max=300 waited=0
    while [ $waited -lt $max ]; do
        local not_ready=$(kubectl get pods --no-headers 2>/dev/null | grep -v "Running\|Completed" | grep -v "^$" | wc -l)
        local running=$(kubectl get pods --no-headers 2>/dev/null | grep "Running" | wc -l)
        if [ "$not_ready" -eq 0 ] && [ "$running" -ge 2 ]; then
            log "pods ready ($running running)"; return 0
        fi
        log "waiting... ($waited/${max}s, running=$running, not_ready=$not_ready)"
        sleep 15; waited=$((waited + 15))
    done
    log "ERROR: timed out"; kubectl get pods; return 1
}

cleanup() {
    log "cleaning up..."
    kubectl delete service generation-service rag-app-service --ignore-not-found=true 2>/dev/null || true
    kubectl delete deployment --all --ignore-not-found=true 2>/dev/null || true
    sleep 5
}

run_experiment() {
    local name=$1 rag_manifest=$2 sched_manifest=$3
    log "=== EXPERIMENT: $name ==="
    cleanup
    if [ -n "$sched_manifest" ]; then
        log "deploying scheduler..."
        kubectl apply -f "$PROJECT_ROOT/$sched_manifest"
        sleep 10
    fi
    log "deploying rag app..."
    kubectl apply -f "$PROJECT_ROOT/$rag_manifest"
    wait_for_pods
    log "waiting 60s for model to load..."
    sleep 60
    kubectl port-forward svc/rag-app-service 8080:8000 &
    PF_PID=$!; sleep 5
    log "health check..."
    local ok=0
    for i in $(seq 1 24); do
        if curl -s http://localhost:8080/health | grep -q "healthy"; then ok=1; break; fi
        sleep 5
    done
    kill $PF_PID 2>/dev/null || true
    if [ "$ok" -ne 1 ]; then
        log "ERROR: health check failed for $name"; return 1
    fi
    log "running $QUERIES queries..."
    python "$PROJECT_ROOT/benchmarks/run_experiments.py" \
        --config "$name" \
        --queries "$QUERIES" \
        --delay "$DELAY" \
        --output "$RESULTS_DIR" \
        --port-forward \
        --skip-wait
    log "experiment $name DONE"
}

log "applying rbac..."
kubectl apply -f "$PROJECT_ROOT/k8s/static-scheduler/rbac.yaml"
kubectl apply -f "$PROJECT_ROOT/k8s/bandit-scheduler/rbac-adaptive.yaml"
sleep 2

run_experiment "baseline-uninformed" "k8s/baseline/01-uninformed-default.yaml" ""
run_experiment "static-ml" "k8s/static-scheduler/rag-deployment.yaml" "k8s/static-scheduler/scheduler-deployment.yaml"
run_experiment "bandit" "k8s/bandit-scheduler/rag-deployment.yaml" "k8s/bandit-scheduler/scheduler-deployment.yaml"
run_experiment "bandit-adaptive" "k8s/bandit-scheduler/rag-deployment-adaptive.yaml" "k8s/bandit-scheduler/scheduler-deployment-adaptive.yaml"

log "=== ALL EXPERIMENTS DONE ==="
python "$PROJECT_ROOT/benchmarks/analyze_results.py" \
    --input "$RESULTS_DIR" \
    --output "$RESULTS_DIR/analysis"
log "=== ANALYSIS DONE ==="
