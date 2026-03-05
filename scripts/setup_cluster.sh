#!/bin/bash
# sets up a k3d cluster with 2 nodes for the scheduler experiments

set -e

CLUSTER_NAME="fyp"
MEMORY=${MEMORY:-8192}
CPUS=${CPUS:-4}

if ! command -v k3d &> /dev/null; then
    echo "k3d not found, install from https://k3d.io"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found"
    exit 1
fi

# delete old cluster if it exists
k3d cluster delete $CLUSTER_NAME 2>/dev/null || true

echo "creating k3d cluster with 2 nodes..."
# 1 server + 1 agent gives us the same topology as the old minikube --nodes=2 setup
# disabling traefik since we don't need an ingress controller
k3d cluster create $CLUSTER_NAME \
    --agents 1 \
    --k3s-arg '--disable=traefik@server:0'

echo "waiting for nodes to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=120s

# label nodes to simulate heterogeneous hardware
# server = cpu_standard (lighter workloads), agent = gpu_accelerated (heavier workloads)
SERVER_NODE="k3d-${CLUSTER_NAME}-server-0"
AGENT_NODE="k3d-${CLUSTER_NAME}-agent-0"

kubectl label node $SERVER_NODE node_type=cpu_standard --overwrite
kubectl label node $AGENT_NODE node_type=gpu_accelerated --overwrite

echo "node labels applied:"
kubectl get nodes --show-labels | grep node_type

# simulate hardware heterogeneity by capping cpu on each node container
# server gets 1 cpu (lighter, retrieval workloads), agent gets 4 cpu (heavier, generation)
# this gives a 1:4 ratio which approximates the relative throughput difference
echo "applying cpu quotas to node containers (1:4 ratio)..."
docker update --cpus="1.0" k3d-${CLUSTER_NAME}-server-0
docker update --cpus="4.0" k3d-${CLUSTER_NAME}-agent-0

echo ""
echo "cluster ready. next steps:"
echo "  1. build images: ./scripts/build_all.sh"
echo "  2. train static model if needed: cd ml-schedulers/static && python train_model.py"
echo "  3. run experiments: ./scripts/run_all_experiments.sh"
