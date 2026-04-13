#!/bin/bash
# run_multiple.sh — Runs run_all.sh N times with fault tolerance.
#
# Features:
#   - Preflight DiskPressure check on CPU node before each attempt
#   - Timestamped parent directory (multi_YYYYMMDD_HHMMSS/)
#   - Temp dir + promote pattern (partial failures don't pollute results)
#   - Bounded retry loop (completed < TARGET && attempt < MAX_ATTEMPTS)
#   - Per-attempt failure log
#   - Aggregated analysis at the end
#
# Usage:
#   ./run_multiple.sh              # 10 runs, max 20 attempts
#   NUM_RUNS=5 ./run_multiple.sh   # 5 runs

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_RUNS="${NUM_RUNS:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
MULTI_DIR="$PROJECT_ROOT/results/multi_$(date +%Y%m%d_%H%M%S)"

# Pass through experiment config env vars to run_all.sh
export QUERIES="${QUERIES:-400}"
export RATE="${RATE:-1.0}"
export RAMP_RATE="${RAMP_RATE:-12.0}"
export CONCURRENCY="${CONCURRENCY:-32}"
export GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-15}"

mkdir -p "$MULTI_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Preflight ──────────────────────────────────────────────────────────────

preflight_check() {
    # Check DiskPressure on all nodes. If any node has DiskPressure=True,
    # the experiment will fail mid-run when pods get evicted.
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

# ─── Main loop ──────────────────────────────────────────────────────────────

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
    log "================================================================"
    log " ATTEMPT $attempt — RUN $run_num / $NUM_RUNS"
    log "================================================================"

    # Preflight: abort attempt early if disk pressure detected
    if ! preflight_check; then
        log "SKIP: DiskPressure detected. Waiting 60s before retry."
        echo "attempt=$attempt run=$run_num status=SKIP reason=DiskPressure timestamp=$(date -Iseconds)" \
            >> "$MULTI_DIR/failure_log.txt"
        sleep 60
        continue
    fi

    # Write to temp dir; promote to final timestamped dir on success
    run_ts=$(date +%Y%m%d_%H%M%S)
    tmp_dir="$MULTI_DIR/.tmp_${run_ts}"
    final_dir="$MULTI_DIR/${run_ts}"
    mkdir -p "$tmp_dir"

    export RESULTS_DIR="$tmp_dir"

    if bash "$PROJECT_ROOT/run_all.sh"; then
        # Success — promote temp to timestamped final dir
        mv "$tmp_dir" "$final_dir"
        completed=$((completed + 1))
        log "Run $run_num SUCCEEDED (attempt $attempt). $completed/$NUM_RUNS complete."
        echo "attempt=$attempt run=$run_num dir=${run_ts} status=OK timestamp=$(date -Iseconds)" \
            >> "$MULTI_DIR/failure_log.txt"
    else
        # Failure — log details, clean up temp
        log "Run $run_num FAILED (attempt $attempt). Cleaning up temp dir."

        # Try to identify which mode failed
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

# ─── Summary ────────────────────────────────────────────────────────────────

log ""
log "================================================================"
log " MULTI-RUN COMPLETE: $completed / $NUM_RUNS runs in $attempt attempts"
log "================================================================"

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
