# generation-service/Dockerfile
# GPU-bound LLM inference service using Ollama
# Model is pulled at container startup via the k8s postStart lifecycle hook.

FROM ollama/ollama:latest

EXPOSE 11434

# Ollama serve is the default entrypoint
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
