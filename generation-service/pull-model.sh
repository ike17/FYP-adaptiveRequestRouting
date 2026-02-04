#!/bin/bash
# generation-service/pull-model.sh
# Pulls the gemma:2b model into Ollama
# This script is called by Kubernetes postStart hook

set -e

echo "Waiting for Ollama server to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama server is ready"
        break
    fi
    echo "Attempt $attempt/$max_attempts - Ollama not ready yet..."
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "ERROR: Ollama server did not become ready in time"
    exit 1
fi

echo "Pulling gemma:2b model..."
ollama pull gemma:2b

echo "Model pull complete. Verifying..."
ollama list

echo "Generation service is ready!"
