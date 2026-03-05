#!/bin/bash
# builds all docker images and loads them into the k3d cluster

set -e

CLUSTER_NAME="fyp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if ! k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
    echo "cluster '$CLUSTER_NAME' not found, run setup_cluster.sh first"
    exit 1
fi

echo "building generation-service..."
cd "$PROJECT_ROOT/generation-service"
docker build -t generation-service:v1 .

echo "building rag-app..."
cd "$PROJECT_ROOT/rag-app"
docker build -t rag-app:v1 .

# static scheduler is optional, skip if model hasn't been trained yet
STATIC_MODEL="$PROJECT_ROOT/ml-schedulers/static/static_scheduler_model.pkl"
if [ -f "$STATIC_MODEL" ]; then
    echo "building static-scheduler..."
    cd "$PROJECT_ROOT/ml-schedulers/static"
    docker build -t static-scheduler:v1 .
else
    echo "skipping static-scheduler (no trained model found)"
    echo "  train it with: cd ml-schedulers/static && python train_model.py"
fi

echo "building bandit-scheduler:v1..."
cd "$PROJECT_ROOT/ml-schedulers/bandit"
docker build -t bandit-scheduler:v1 .

echo "building bandit-scheduler:v2-adaptive..."
docker build -f Dockerfile.adaptive -t bandit-scheduler:v2-adaptive .

# k3d nodes don't share the host docker daemon so we need to import images explicitly
echo "importing images into k3d cluster..."
k3d image import generation-service:v1 rag-app:v1 bandit-scheduler:v1 bandit-scheduler:v2-adaptive -c $CLUSTER_NAME

if [ -f "$STATIC_MODEL" ]; then
    k3d image import static-scheduler:v1 -c $CLUSTER_NAME
fi

echo "done"
docker images | grep -E "(generation-service|rag-app|static-scheduler|bandit-scheduler)"
