#!/bin/bash
set -euo pipefail

if [ -z "${GPU_NODE:-}" ]; then
    echo "ERROR: GPU_NODE is not set."
    echo "  Usage: GPU_NODE=192.168.0.xx ./scripts/build_all.sh"
    exit 1
fi

SSH_USER="${SSH_USER:-$(whoami)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Checking kubectl connectivity..."
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "ERROR: kubectl cannot reach the cluster. Check KUBECONFIG."
    exit 1
fi
log "  OK"

log "Building generation-service:v1..."
cd "$PROJECT_ROOT/generation-service"
docker build -t generation-service:v1 .

STATIC_MODEL="$PROJECT_ROOT/ml-training/gateway_model.pkl"
if [ -f "$STATIC_MODEL" ]; then
    log "Copying gateway_model.pkl into smart-gateway/model/..."
    mkdir -p "$PROJECT_ROOT/smart-gateway/model"
    cp "$STATIC_MODEL" "$PROJECT_ROOT/smart-gateway/model/gateway_model.pkl"
else
    log "NOTE: ml-training/gateway_model.pkl not found — static routing mode will fail."
    log "      Train it first: cd ml-training && python generate_dataset.py && python train_model.py"
fi

log "Building smart-gateway:v1..."
cd "$PROJECT_ROOT/smart-gateway"
docker build -t smart-gateway:v1 .

import_image() {
    local IMAGE="$1"
    log "Importing $IMAGE into k3s on local node (ProDesk)..."
    docker save "$IMAGE" | sudo k3s ctr images import -
    log "  OK (local)"

    log "Importing $IMAGE into k3s on GPU node ($SSH_USER@$GPU_NODE)..."
    docker save "$IMAGE" | ssh "${SSH_USER}@${GPU_NODE}" sudo k3s ctr images import -
    log "  OK (remote)"
}

import_image "generation-service:v1"
import_image "smart-gateway:v1"

log "Images loaded on local node:"
sudo k3s ctr images list 2>/dev/null | grep -E "(generation-service|smart-gateway)" || echo "  (none found)"

log "Images loaded on GPU node ($GPU_NODE):"
ssh "${SSH_USER}@${GPU_NODE}" sudo k3s ctr images list 2>/dev/null \
    | grep -E "(generation-service|smart-gateway)" || echo "  (none found)"

echo ""
echo "Build complete"
echo "  generation-service:v1 — imported on both nodes"
echo "  smart-gateway:v1      — imported on both nodes"
echo ""
echo "Next: kubectl apply -f k8s/"
echo "  or:  ./run_all.sh"
