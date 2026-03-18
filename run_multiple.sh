#!/bin/bash
set -e

PROJECT_ROOT="/c/Users/shaik/Nextcloud/Lectures/FYP/fyp-rag-scheduler/fyp-rag-scheduler"
NUM_RUNS=10

echo "Starting $NUM_RUNS experimental runs..."

for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "================================================================"
    echo " RUN $i / $NUM_RUNS "
    echo "================================================================"
    
    export RESULTS_DIR="$PROJECT_ROOT/results/run_$i"
    
    # Run the standard experiments for this iteration
    bash "$PROJECT_ROOT/run_all.sh"
    
    echo "Run $i completed. Results saved to $RESULTS_DIR"
done

echo ""
echo "================================================================"
echo " ALL $NUM_RUNS RUNS COMPLETED "
echo "================================================================"
echo "Running aggregated analysis..."

python3 "$PROJECT_ROOT/benchmarks/analyze_multiple.py" \
    --input "$PROJECT_ROOT/results" \
    --runs "$NUM_RUNS" \
    --output "$PROJECT_ROOT/results/aggregated_analysis"

echo "Aggregated analysis done. Check results/aggregated_analysis/"
