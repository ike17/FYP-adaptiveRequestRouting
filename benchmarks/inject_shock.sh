#!/bin/bash
# benchmarks/inject_shock.sh
# Toggle the physical CPU starvation shock on the generation node.
# Usage:
#   bash benchmarks/inject_shock.sh start   -- deploy cpu-stress pod to agent node
#   bash benchmarks/inject_shock.sh stop    -- remove cpu-stress pod

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="$SCRIPT_DIR/../k8s/shock/cpu-stress.yaml"

if [ "$1" == "start" ]; then
    echo "[shock] deploying cpu-stress pod to k3d-fyp-agent-0..."
    kubectl apply -f "$YAML"
    echo "[shock] stress active — generation node CPU is now contested"
elif [ "$1" == "stop" ]; then
    echo "[shock] removing cpu-stress pod..."
    kubectl delete -f "$YAML" --ignore-not-found=true 2>/dev/null || kubectl delete pod cpu-stress --ignore-not-found=true 2>/dev/null || true
    echo "[shock] stress removed"
else
    echo "Usage: $0 {start|stop}"
    exit 1
fi
