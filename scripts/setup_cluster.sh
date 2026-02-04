#!/bin/bash
# scripts/setup_cluster.sh
# Initialize Minikube cluster with GPU support and node labels

set -e

echo "============================================================"
echo "FYP RAG SCHEDULER - CLUSTER SETUP"
echo "============================================================"

# Configuration
MEMORY=${MEMORY:-8192}
CPUS=${CPUS:-4}
DRIVER=${DRIVER:-docker}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi
print_status "Docker found"

if ! command -v minikube &> /dev/null; then
    print_error "Minikube not found. Please install Minikube first."
    echo "  curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64"
    echo "  sudo install minikube-linux-amd64 /usr/local/bin/minikube"
    exit 1
fi
print_status "Minikube found"

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl not found. Please install kubectl first."
    exit 1
fi
print_status "kubectl found"

# Check for NVIDIA GPU (optional)
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected"
    GPU_FLAG="--gpus=all"
else
    print_warning "No NVIDIA GPU detected. GPU workloads will run on CPU (slower)."
fi

# Stop existing cluster if running
echo ""
echo "Cleaning up existing cluster..."
minikube delete 2>/dev/null || true
print_status "Cleanup complete"

# Start Minikube with 2 nodes
echo ""
echo "Starting Minikube cluster..."
echo "  Memory: ${MEMORY}MB"
echo "  CPUs: ${CPUS}"
echo "  Driver: ${DRIVER}"
echo "  GPU: ${GPU_FLAG:-none}"

minikube start \
    --nodes=2 \
    --driver=${DRIVER} \
    ${GPU_FLAG} \
    --memory=${MEMORY} \
    --cpus=${CPUS} \
    --kubernetes-version=v1.28.0

print_status "Minikube cluster started"

# Wait for nodes to be ready
echo ""
echo "Waiting for nodes to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s
print_status "All nodes ready"

# Label nodes for heterogeneous scheduling
echo ""
echo "Labeling nodes..."

# Node 1 (control plane): CPU standard
kubectl label node minikube node_type=cpu_standard --overwrite
print_status "minikube labeled as cpu_standard"

# Node 2: GPU accelerated
kubectl label node minikube-m02 node_type=gpu_accelerated --overwrite
print_status "minikube-m02 labeled as gpu_accelerated"

# Verify labels
echo ""
echo "Node labels:"
kubectl get nodes --show-labels | grep node_type

# Install NVIDIA device plugin if GPU available
if [ -n "$GPU_FLAG" ]; then
    echo ""
    echo "Installing NVIDIA device plugin..."
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml 2>/dev/null || true
    print_status "NVIDIA device plugin installed"
    
    # Wait for device plugin
    sleep 10
    
    # Verify GPU is available
    echo ""
    echo "Checking GPU availability..."
    GPU_COUNT=$(kubectl get nodes -o json | grep -c "nvidia.com/gpu" || echo "0")
    if [ "$GPU_COUNT" -gt 0 ]; then
        print_status "GPU resources detected in cluster"
    else
        print_warning "GPU resources not yet visible. They may appear after device plugin initializes."
    fi
fi

# Display cluster info
echo ""
echo "============================================================"
echo "CLUSTER SETUP COMPLETE"
echo "============================================================"
echo ""
kubectl get nodes -o wide
echo ""
echo "Next steps:"
echo "  1. Run ./scripts/build_all.sh to build Docker images"
echo "  2. Train the ML model: cd ml-schedulers/static && python train_model.py"
echo "  3. Deploy and run experiments"
echo ""
