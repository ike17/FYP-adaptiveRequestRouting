#!/bin/bash
# scripts/build_all.sh
# Build all Docker images and load them into Minikube

set -e

echo "============================================================"
echo "FYP RAG SCHEDULER - BUILD ALL IMAGES"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Check Minikube is running
if ! minikube status | grep -q "Running"; then
    print_error "Minikube is not running. Start it first with ./scripts/setup_cluster.sh"
    exit 1
fi

# Configure Docker to use Minikube's Docker daemon
echo ""
echo "Configuring Docker to use Minikube's daemon..."
eval $(minikube docker-env)
print_status "Docker configured for Minikube"

# Build generation-service
echo ""
echo "============================================================"
echo "Building generation-service (Ollama + LLM)"
echo "============================================================"
cd "$PROJECT_ROOT/generation-service"
docker build -t generation-service:v1 .
print_status "generation-service:v1 built"

# Build rag-app
echo ""
echo "============================================================"
echo "Building rag-app (FastAPI + ChromaDB)"
echo "============================================================"
cd "$PROJECT_ROOT/rag-app"
docker build -t rag-app:v1 .
print_status "rag-app:v1 built"

# Check if static scheduler model exists
STATIC_MODEL="$PROJECT_ROOT/ml-schedulers/static/static_scheduler_model.pkl"
if [ -f "$STATIC_MODEL" ]; then
    echo ""
    echo "============================================================"
    echo "Building static-scheduler"
    echo "============================================================"
    cd "$PROJECT_ROOT/ml-schedulers/static"
    docker build -t static-scheduler:v1 .
    print_status "static-scheduler:v1 built"
else
    print_warning "Static scheduler model not found. Train it first:"
    echo "  cd ml-schedulers/static"
    echo "  pip install -r requirements.txt"
    echo "  python generate_dataset.py"
    echo "  python train_model.py"
    echo ""
    echo "Then re-run this script to build the static-scheduler image."
fi

# Build bandit-scheduler
echo ""
echo "============================================================"
echo "Building bandit-scheduler"
echo "============================================================"
cd "$PROJECT_ROOT/ml-schedulers/bandit"
docker build -t bandit-scheduler:v1 .
print_status "bandit-scheduler:v1 built"

# Verify images
echo ""
echo "============================================================"
echo "BUILT IMAGES"
echo "============================================================"
docker images | grep -E "(generation-service|rag-app|static-scheduler|bandit-scheduler)" | head -10

echo ""
echo "============================================================"
echo "BUILD COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Deploy baseline: kubectl apply -f k8s/baseline/03-manual-optimal.yaml"
echo "  2. Run experiments: ./scripts/run_all_experiments.sh"
echo ""
