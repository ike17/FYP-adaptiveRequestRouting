# generation-service/Dockerfile
# GPU-bound LLM inference service using Ollama

FROM ollama/ollama:latest

# Copy model pull script
COPY pull-model.sh /pull-model.sh
RUN chmod +x /pull-model.sh

# Note: Model will be pulled at container startup via postStart hook
# This avoids baking the model into the image (saves space, allows flexibility)

EXPOSE 11434

# Ollama serve is the default entrypoint
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
