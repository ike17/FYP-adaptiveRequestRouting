#!/bin/bash
# scripts/run_all_experiments.sh
# Automated experiment runner for all scheduler configurations

set -e

echo "============================================================"
echo "FYP RAG SCHEDULER - AUTOMATED EXPERIMENTS"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
QUERIES=${QUERIES:-100}
DELAY=${DELAY:-0.5}
RESULTS_DIR="$PROJECT_ROOT/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to wait for pods to be ready
wait_for_pods() {
    echo "Waiting for pods to be ready..."
    local max_wait=300
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        # Check if all pods are ready
        local not_ready=$(kubectl get pods --no-headers 2>/dev/null | grep -v "Running\|Completed" | wc -l)
        
        if [ "$not_ready" -eq 0 ]; then
            local running=$(kubectl get pods --no-headers 2>/dev/null | grep "Running" | wc -l)
            if [ "$running" -ge 2 ]; then
                print_status "All pods ready ($running running)"
                return 0
            fi
        fi
        
        echo "  Waiting... ($waited/$max_wait seconds)"
        sleep 10
        waited=$((waited + 10))
    done
    
    print_error "Timeout waiting for pods"
    kubectl get pods
    return 1
}

# Function to clean up deployments
cleanup() {
    echo "Cleaning up previous deployments..."
    kubectl delete deployment --all --ignore-not-found=true 2>/dev/null || true
    kubectl delete service generation-service rag-app-service --ignore-not-found=true 2>/dev/null || true
    sleep 5
    print_status "Cleanup complete"
}

# Function to run single experiment
run_experiment() {
    local config_name=$1
    local manifest=$2
    local scheduler_manifest=$3
    
    print_header "EXPERIMENT: $config_name"
    
    # Cleanup
    cleanup
    
    # Deploy scheduler if needed
    if [ -n "$scheduler_manifest" ]; then
        echo "Deploying scheduler..."
        kubectl apply -f "$PROJECT_ROOT/$scheduler_manifest"
        sleep 10
    fi
    
    # Deploy RAG application
    echo "Deploying RAG application..."
    kubectl apply -f "$PROJECT_ROOT/$manifest"
    
    # Wait for pods
    if ! wait_for_pods; then
        print_error "Failed to start pods for $config_name"
        kubectl get pods
        kubectl describe pods
        return 1
    fi
    
    # Extra wait for model to load
    echo "Waiting for LLM model to load (this may take a few minutes)..."
    sleep 60
    
    # Get RAG URL
    local rag_url=$(minikube service rag-app-service --url 2>/dev/null | head -1)
    if [ -z "$rag_url" ]; then
        print_error "Could not get RAG service URL"
        return 1
    fi
    echo "RAG URL: $rag_url"
    
    # Wait for service to be healthy
    echo "Waiting for service health check..."
    local health_wait=0
    while [ $health_wait -lt 120 ]; do
        if curl -s "$rag_url/health" | grep -q "healthy"; then
            print_status "Service is healthy"
            break
        fi
        sleep 5
        health_wait=$((health_wait + 5))
    done
    
    # Run benchmark
    echo "Running $QUERIES queries..."
    cd "$PROJECT_ROOT/benchmarks"
    python run_experiments.py \
        --config "$config_name" \
        --queries "$QUERIES" \
        --delay "$DELAY" \
        --output "$RESULTS_DIR" \
        --url "$rag_url" \
        --skip-wait
    
    print_status "Experiment $config_name complete"
}

# Main execution
print_header "EXPERIMENT CONFIGURATION"
echo "Queries per experiment: $QUERIES"
echo "Delay between queries: ${DELAY}s"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check prerequisites
if ! minikube status | grep -q "Running"; then
    print_error "Minikube is not running"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -r "$PROJECT_ROOT/benchmarks/requirements.txt"

# ============================================================
# RUN EXPERIMENTS
# ============================================================

# Experiment 1: Baseline Uninformed
run_experiment "baseline-uninformed" "k8s/baseline/01-uninformed-default.yaml" ""

# Experiment 2: Baseline Informed
run_experiment "baseline-informed" "k8s/baseline/02-informed-default.yaml" ""

# Experiment 3: Baseline Optimal (Manual)
run_experiment "baseline-optimal" "k8s/baseline/03-manual-optimal.yaml" ""

# Experiment 4: Static ML Scheduler
# First deploy RBAC and scheduler
if docker images | grep -q "static-scheduler"; then
    kubectl apply -f "$PROJECT_ROOT/k8s/static-scheduler/rbac.yaml"
    run_experiment "static-ml" "k8s/static-scheduler/rag-deployment.yaml" "k8s/static-scheduler/scheduler-deployment.yaml"
else
    print_warning "Skipping static-ml experiment - image not built"
    print_warning "Run: cd ml-schedulers/static && python train_model.py && ./scripts/build_all.sh"
fi

# Experiment 5: Bandit Scheduler
kubectl apply -f "$PROJECT_ROOT/k8s/bandit-scheduler/rbac.yaml"
run_experiment "bandit-adaptive" "k8s/bandit-scheduler/rag-deployment.yaml" "k8s/bandit-scheduler/scheduler-deployment.yaml"

# ============================================================
# ANALYZE RESULTS
# ============================================================

print_header "ANALYZING RESULTS"

cd "$PROJECT_ROOT/benchmarks"
python analyze_results.py --input "$RESULTS_DIR" --output "$RESULTS_DIR/analysis"

print_header "ALL EXPERIMENTS COMPLETE"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Analysis saved to: $RESULTS_DIR/analysis"
echo ""
echo "Generated files:"
ls -la "$RESULTS_DIR/analysis/" 2>/dev/null || echo "  (analysis directory empty)"
echo ""
