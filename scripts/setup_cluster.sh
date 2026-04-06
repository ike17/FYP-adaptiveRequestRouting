#!/bin/bash
# scripts/setup_cluster.sh
#
# Configures an existing bare-metal k3s cluster for the FYP experiment.
# k3s must already be installed and both nodes must be joined before running this.
#
# Cluster topology:
#   CPU node (k3s server) — ProDesk i5-9500, 192.168.0.20, user: prox
#   GPU node (k3s agent)  — Laptop RTX 3070
#
# What this script does:
#   1. Verifies kubectl works and both nodes are Ready
#   2. Labels nodes with node_type=cpu_standard / gpu_accelerated
#   3. Applies k8s/00-nvidia-runtime-class.yaml (RuntimeClass "nvidia")
#   4. Applies the NVIDIA device plugin DaemonSet (v0.17.1)
#   5. Patches the device plugin DaemonSet to set runtimeClassName: nvidia
#   6. Waits for the device plugin to become Ready
#   7. Prints node summary including nvidia.com/gpu capacity
#
# Usage:
#   ./scripts/setup_cluster.sh
#
# Optional env vars:
#   CPU_NODE_NAME — hostname of the k3s server node (auto-detected if unset)
#   GPU_NODE_NAME — hostname of the k3s agent node  (auto-detected if unset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEVICE_PLUGIN_URL="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Verify cluster connectivity ─────────────────────────────────────────────

log "Checking kubectl connectivity..."
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "ERROR: kubectl cannot reach the cluster."
    echo "  Make sure KUBECONFIG is set or ~/.kube/config is valid."
    exit 1
fi
log "  OK"

# ─── Discover node names ──────────────────────────────────────────────────────

ALL_NODES=$(kubectl get nodes -o jsonpath='{.items[*].metadata.name}')
NODE_COUNT=$(echo "$ALL_NODES" | wc -w)

if [ "$NODE_COUNT" -lt 2 ]; then
    echo "ERROR: Expected at least 2 nodes, found: $NODE_COUNT"
    echo "  Nodes: $ALL_NODES"
    echo "  Make sure both nodes have joined the cluster and are Ready."
    exit 1
fi

log "Found nodes: $ALL_NODES"

# The k3s server is the node that runs the control plane.
# Detect it by looking for the 'node-role.kubernetes.io/master' or 'control-plane' label.
if [ -z "${CPU_NODE_NAME:-}" ]; then
    CPU_NODE_NAME=$(kubectl get nodes \
        -l 'node-role.kubernetes.io/control-plane' \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -z "$CPU_NODE_NAME" ]; then
        # Fallback: try old 'master' label
        CPU_NODE_NAME=$(kubectl get nodes \
            -l 'node-role.kubernetes.io/master' \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    fi
    if [ -z "$CPU_NODE_NAME" ]; then
        echo "ERROR: Could not auto-detect the CPU (server) node."
        echo "  Set CPU_NODE_NAME=<hostname> before running."
        exit 1
    fi
fi

if [ -z "${GPU_NODE_NAME:-}" ]; then
    # GPU node = any node that isn't the CPU node
    GPU_NODE_NAME=$(kubectl get nodes \
        -o jsonpath='{.items[*].metadata.name}' \
        | tr ' ' '\n' | grep -v "^${CPU_NODE_NAME}$" | head -1)
    if [ -z "$GPU_NODE_NAME" ]; then
        echo "ERROR: Could not auto-detect the GPU (agent) node."
        echo "  Set GPU_NODE_NAME=<hostname> before running."
        exit 1
    fi
fi

log "CPU node (server): $CPU_NODE_NAME"
log "GPU node (agent):  $GPU_NODE_NAME"

# Wait for both nodes to be Ready
for NODE in "$CPU_NODE_NAME" "$GPU_NODE_NAME"; do
    log "Waiting for node $NODE to be Ready..."
    kubectl wait node "$NODE" --for=condition=Ready --timeout=120s
    log "  $NODE Ready"
done

# ─── Label nodes ─────────────────────────────────────────────────────────────

log "Labelling CPU node: node_type=cpu_standard"
kubectl label node "$CPU_NODE_NAME" node_type=cpu_standard --overwrite

log "Labelling GPU node: node_type=gpu_accelerated"
kubectl label node "$GPU_NODE_NAME" node_type=gpu_accelerated --overwrite

# ─── RuntimeClass ─────────────────────────────────────────────────────────────

log "Applying NVIDIA RuntimeClass..."
kubectl apply -f "$PROJECT_ROOT/k8s/00-nvidia-runtime-class.yaml"

# ─── NVIDIA device plugin ────────────────────────────────────────────────────

log "Applying NVIDIA device plugin DaemonSet (v0.17.1)..."
kubectl apply -f "$DEVICE_PLUGIN_URL"

# Give the DaemonSet a moment to be created before patching
sleep 3

log "Patching device plugin DaemonSet to use runtimeClassName: nvidia..."
kubectl -n kube-system patch ds nvidia-device-plugin-daemonset \
    --type='json' \
    -p='[{"op":"add","path":"/spec/template/spec/runtimeClassName","value":"nvidia"}]'

log "Waiting for device plugin DaemonSet to be Ready on GPU node..."
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s

# ─── Verify GPU resource is visible ──────────────────────────────────────────

log "Waiting up to 60s for GPU resource to register on node $GPU_NODE_NAME..."
for i in $(seq 1 12); do
    GPU_CAP=$(kubectl get node "$GPU_NODE_NAME" \
        -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
    if [ "${GPU_CAP:-0}" != "0" ] && [ -n "${GPU_CAP:-}" ]; then
        log "  nvidia.com/gpu capacity on $GPU_NODE_NAME: $GPU_CAP"
        break
    fi
    sleep 5
    log "  waiting for GPU resource... ($((i*5))s/60s)"
done

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "======================================"
echo "NODE SUMMARY"
echo "======================================"
kubectl get nodes -o wide
echo ""
echo "======================================"
echo "GPU CAPACITY"
echo "======================================"
kubectl describe nodes | grep -A5 "Capacity:" | grep -E "(nvidia|gpu)" || echo "  (no GPU resource visible yet)"
echo ""
echo "======================================"
echo "Setup complete!"
echo "  CPU node: $CPU_NODE_NAME  (node_type=cpu_standard)"
echo "  GPU node: $GPU_NODE_NAME  (node_type=gpu_accelerated)"
echo ""
echo "Next: ./scripts/build_all.sh GPU_NODE=<laptop-ip>"
echo "======================================"
