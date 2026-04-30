#!/bin/bash
# Repeats run_all.sh N times with retries, then aggregates results across runs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_RUNS="${NUM_RUNS:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
MULTI_DIR="$PROJECT_ROOT/results/multi_$(date +%Y%m%d_%H%M%S)"

export QUERIES="${QUERIES:-400}"
export RATE="${RATE:-1.0}"
export RAMP_RATE="${RAMP_RATE:-12.0}"
export CONCURRENCY="${CONCURRENCY:-32}"
export GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-15}"

mkdir -p "$MULTI_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

preflight_check() {
    log "preflight: checking node conditions..."
    local pressure
    pressure=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{range .status.conditions[?(@.type=="DiskPressure")]}{.status}{end}{"\n"}{end}' 2>/dev/null)

    local failed=0
    while IFS= read -r line; do
        local node status
        node=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $2}')
        if [ -z "$node" ]; then continue; fi
        if [ "$status" = "True" ]; then
            log "  FAIL: $node has DiskPressure=True"
            failed=1
        else
            log "  OK: $node DiskPressure=$status"
        fi
    done <<< "$pressure"

    return $failed
}

log "Starting multi-run experiment suite"
log "  Target runs  : $NUM_RUNS"
log "  Max attempts : $MAX_ATTEMPTS"
log "  Output       : $MULTI_DIR"
log "  Config       : Q=$QUERIES rate=$RATE ramp=$RAMP_RATE conc=$CONCURRENCY timeout=${GENERATION_TIMEOUT}s"

completed=0
attempt=0

while [ "$completed" -lt "$NUM_RUNS" ] && [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    run_num=$((completed + 1))

    log ""
    log "Attempt $attempt — Run $run_num / $NUM_RUNS"

    if ! preflight_check; then
        log "SKIP: DiskPressure detected. Waiting 60s before retry."
        echo "attempt=$attempt run=$run_num status=SKIP reason=DiskPressure timestamp=$(date -Iseconds)" \
            >> "$MULTI_DIR/failure_log.txt"
        sleep 60
        continue
    fi

    run_ts=$(date +%Y%m%d_%H%M%S)
    tmp_dir="$MULTI_DIR/.tmp_${run_ts}"
    final_dir="$MULTI_DIR/${run_ts}"
    mkdir -p "$tmp_dir"

    export RESULTS_DIR="$tmp_dir"

    if bash "$PROJECT_ROOT/run_all.sh"; then
        mv "$tmp_dir" "$final_dir"
        completed=$((completed + 1))
        log "Run $run_num SUCCEEDED (attempt $attempt). $completed/$NUM_RUNS complete."
        echo "attempt=$attempt run=$run_num dir=${run_ts} status=OK timestamp=$(date -Iseconds)" \
            >> "$MULTI_DIR/failure_log.txt"
    else
        log "Run $run_num FAILED (attempt $attempt). Cleaning up temp dir."

        local_modes=""
        for mode in baseline least_in_flight static bandit_plain bandit_regime adaptive; do
            if [ ! -f "$tmp_dir/${mode}_summary.json" ]; then
                local_modes="${local_modes}${local_modes:+,}$mode"
            fi
        done
        [ -z "$local_modes" ] && local_modes="unknown"

        echo "attempt=$attempt run=$run_num status=FAIL missing_modes=$local_modes timestamp=$(date -Iseconds)" \
            >> "$MULTI_DIR/failure_log.txt"

        rm -rf "$tmp_dir"

        log "Waiting 30s before retry..."
        sleep 30
    fi
done

log ""
log "Multi-run complete: $completed / $NUM_RUNS runs in $attempt attempts"

if [ "$completed" -lt "$NUM_RUNS" ]; then
    log "WARNING: only $completed runs completed (target was $NUM_RUNS)"
fi

if [ "$completed" -gt 0 ]; then
    log "Running aggregated analysis..."
    python3 "$PROJECT_ROOT/benchmarks/analyze_multiple.py" \
        --input "$MULTI_DIR" \
        --runs "$completed" \
        --output "$MULTI_DIR/aggregated_analysis"
    log "Aggregated analysis: $MULTI_DIR/aggregated_analysis/"
fi

log "Failure log: $MULTI_DIR/failure_log.txt"
log "Done."
