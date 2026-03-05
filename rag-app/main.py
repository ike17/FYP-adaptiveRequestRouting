# main.py
# fastapi app that handles the rag pipeline - retrieval then generation
# the /metrics endpoint is polled by the bandit scheduler for reward signals

import os
import time
import logging
from datetime import datetime
from typing import Optional
from collections import deque

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests

from vectordb import VectorDB

GENERATION_SERVICE_URL = os.getenv(
    "GENERATION_SERVICE_URL",
    "http://generation-service:11434/api/generate"
)
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "60"))
METRICS_HISTORY_SIZE = int(os.getenv("METRICS_HISTORY_SIZE", "500"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application", version="1.0.0")

logger.info("initialising vector database...")
vector_db = VectorDB()
logger.info("vector database ready")

# rolling window of query metrics - the scheduler polls this
metrics_history = deque(maxlen=METRICS_HISTORY_SIZE)
query_counter = 0
startup_time = time.time()


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    include_context: bool = Field(True)
    top_k: int = Field(3, ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is Kubernetes scheduling?",
                "include_context": True,
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    query_id: int
    response: str
    context_docs: list[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    timestamp: str


class MetricsResponse(BaseModel):
    total_queries: int
    history: list[dict]
    avg_retrieval_ms: Optional[float]
    avg_generation_ms: Optional[float]
    avg_total_ms: Optional[float]
    p50_total_ms: Optional[float]
    p95_total_ms: Optional[float]
    p99_total_ms: Optional[float]


class HealthResponse(BaseModel):
    status: str
    service: str
    vector_db_docs: int
    generation_service_url: str
    uptime_seconds: float


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="rag-app",
        vector_db_docs=vector_db.collection.count(),
        generation_service_url=GENERATION_SERVICE_URL,
        uptime_seconds=round(time.time() - startup_time, 2)
    )


@app.get("/ready")
async def readiness_check():
    try:
        response = requests.get(
            GENERATION_SERVICE_URL.replace("/api/generate", "/api/tags"),
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "ready", "generation_service": "reachable"}
    except Exception as e:
        logger.warning(f"generation service not reachable: {e}")
        raise HTTPException(status_code=503, detail=f"generation service not ready: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest):
    """runs the full rag pipeline: retrieve context then generate a response"""
    global query_counter
    query_counter += 1
    current_query_id = query_counter

    total_start = time.time()
    timestamp = datetime.utcnow().isoformat()

    logger.info(f"[query {current_query_id}] starting: {request.prompt[:50]}...")

    try:
        # stage 1: retrieval (cpu-bound)
        retrieval_start = time.time()
        context_docs = vector_db.retrieve(request.prompt, top_k=request.top_k)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        logger.info(f"[query {current_query_id}] retrieval: {retrieval_time_ms:.2f}ms, {len(context_docs)} docs")

        # stage 2: generation (sent to ollama over the network)
        generation_start = time.time()

        if request.include_context and context_docs:
            context_str = "\n".join(f"- {doc}" for doc in context_docs)
            augmented_prompt = f"""Use the following context to answer the question.

Context:
{context_str}

Question: {request.prompt}

Answer:"""
        else:
            augmented_prompt = request.prompt

        payload = {
            "model": "gemma:2b",
            "prompt": augmented_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 256,
                "top_p": 0.9
            }
        }

        logger.info(f"[query {current_query_id}] sending to generation service...")

        try:
            response = requests.post(
                GENERATION_SERVICE_URL,
                json=payload,
                timeout=GENERATION_TIMEOUT
            )
            response.raise_for_status()
            generation_result = response.json()
            generated_text = generation_result.get("response", "")

        except requests.exceptions.Timeout:
            logger.error(f"[query {current_query_id}] generation timeout after {GENERATION_TIMEOUT}s")
            raise HTTPException(status_code=504, detail=f"generation service timeout after {GENERATION_TIMEOUT}s")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[query {current_query_id}] cannot connect to generation service: {e}")
            raise HTTPException(status_code=503, detail="generation service unavailable")
        except requests.exceptions.RequestException as e:
            logger.error(f"[query {current_query_id}] generation request failed: {e}")
            raise HTTPException(status_code=502, detail=f"generation service error: {str(e)}")

        generation_time_ms = (time.time() - generation_start) * 1000
        logger.info(f"[query {current_query_id}] generation: {generation_time_ms:.2f}ms")

        total_time_ms = (time.time() - total_start) * 1000
        logger.info(f"[query {current_query_id}] total: {total_time_ms:.2f}ms (retrieval: {retrieval_time_ms:.2f}ms, generation: {generation_time_ms:.2f}ms)")

        metrics_history.append({
            "query_id": current_query_id,
            "timestamp": timestamp,
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "generation_time_ms": round(generation_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "success": True
        })

        return QueryResponse(
            query_id=current_query_id,
            response=generated_text,
            context_docs=context_docs,
            retrieval_time_ms=round(retrieval_time_ms, 2),
            generation_time_ms=round(generation_time_ms, 2),
            total_time_ms=round(total_time_ms, 2),
            timestamp=timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[query {current_query_id}] pipeline error: {e}", exc_info=True)

        total_time_ms = (time.time() - total_start) * 1000
        metrics_history.append({
            "query_id": current_query_id,
            "timestamp": timestamp,
            "retrieval_time_ms": 0,
            "generation_time_ms": 0,
            "total_time_ms": round(total_time_ms, 2),
            "success": False,
            "error": str(e)
        })

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """latency history - the bandit scheduler polls this for reward signals"""
    history_list = list(metrics_history)

    if history_list:
        successful = [m for m in history_list if m.get("success", True)]

        if successful:
            retrieval_times = [m["retrieval_time_ms"] for m in successful]
            generation_times = [m["generation_time_ms"] for m in successful]
            total_times = [m["total_time_ms"] for m in successful]

            sorted_totals = sorted(total_times)
            n = len(sorted_totals)

            return MetricsResponse(
                total_queries=query_counter,
                history=history_list,
                avg_retrieval_ms=round(sum(retrieval_times) / len(retrieval_times), 2),
                avg_generation_ms=round(sum(generation_times) / len(generation_times), 2),
                avg_total_ms=round(sum(total_times) / len(total_times), 2),
                p50_total_ms=round(sorted_totals[int(n * 0.50)], 2),
                p95_total_ms=round(sorted_totals[min(int(n * 0.95), n-1)], 2),
                p99_total_ms=round(sorted_totals[min(int(n * 0.99), n-1)], 2)
            )

    return MetricsResponse(
        total_queries=query_counter,
        history=history_list,
        avg_retrieval_ms=None,
        avg_generation_ms=None,
        avg_total_ms=None,
        p50_total_ms=None,
        p95_total_ms=None,
        p99_total_ms=None
    )


@app.delete("/metrics")
async def clear_metrics():
    """clear metrics between experiment runs"""
    global query_counter
    metrics_history.clear()
    query_counter = 0
    return {"status": "cleared", "message": "metrics history reset"}


@app.get("/")
async def root():
    return {
        "service": "RAG Application",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "run the rag pipeline",
            "GET /metrics": "latency history",
            "GET /health": "health check",
            "GET /ready": "readiness check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
