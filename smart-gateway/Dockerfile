FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer embedding model so the first request
# isn't slow waiting for a 90 MB download inside the cluster.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download the ChromaDB ONNX embedding model so the liveness probe isn't
# killed while a fresh container fetches the 79 MB model at runtime.
RUN python -c "import chromadb; c = chromadb.Client(); col = c.create_collection('warmup'); col.add(documents=['warmup'], ids=['0'])"

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Single worker — in-memory state (inflight counters, bandit posteriors) must
# not be split across processes. See k8s/03-gateway-app.yaml (replicas: 1).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
