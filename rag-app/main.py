"""
rag-app/main.py
RAG Application - Monolithic service handling API and retrieval.
Communicates with generation-service for LLM inference.

Key Features:
- POST /query: Execute full RAG pipeline
- GET /metrics: Return latency history for scheduler feedback
- GET /health: Kubernetes health check
"""

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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Generation service URL (Kubernetes DNS)
GENERATION_SERVICE_URL = os.getenv(
    "GENERATION_SERVICE_URL",
    "http://generation-service:11434/api/generate"
)

# Generation timeout in seconds
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT", "60"))

# Metrics history size
METRICS_HISTORY_SIZE = int(os.getenv("METRICS_HISTORY_SIZE", "500"))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="RAG Application",
    description="Retrieval-Augmented Generation service for Kubernetes scheduling experiments",
    version="1.0.0"
)

# Initialize vector database (CPU-bound initialization)
logger.info("Initializing vector database...")
vector_db = VectorDB()
logger.info("Vector database ready")

# =============================================================================
# METRICS STORAGE
# =============================================================================

# Thread-safe metrics storage using deque
metrics_history = deque(maxlen=METRICS_HISTORY_SIZE)
query_counter = 0

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Input model for RAG query."""
    prompt: str = Field(..., description="User question or query", min_length=1)
    include_context: bool = Field(True, description="Whether to include retrieved context")
    top_k: int = Field(3, description="Number of documents to retrieve", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is Kubernetes scheduling?",
                "include_context": True,
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    """Output model for RAG query."""
    query_id: int = Field(..., description="Unique query identifier")
    response: str = Field(..., description="Generated response from LLM")
    context_docs: list[str] = Field(..., description="Retrieved context documents")
    retrieval_time_ms: float = Field(..., description="Time spent on retrieval stage")
    generation_time_ms: float = Field(..., description="Time spent on generation stage")
    total_time_ms: float = Field(..., description="Total end-to-end latency")
    timestamp: str = Field(..., description="ISO timestamp of query")


class MetricsResponse(BaseModel):
    """Output model for metrics endpoint."""
    total_queries: int
    history: list[dict]
    avg_retrieval_ms: Optional[float]
    avg_generation_ms: Optional[float]
    avg_total_ms: Optional[float]
    p50_total_ms: Optional[float]
    p95_total_ms: Optional[float]
    p99_total_ms: Optional[float]


class HealthResponse(BaseModel):
    """Output model for health check."""
    status: str
    service: str
    vector_db_docs: int
    generation_service_url: str
    uptime_seconds: float


# Track startup time
startup_time = time.time()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Kubernetes probes.
    Returns service status and configuration info.
    """
    return HealthResponse(
        status="healthy",
        service="rag-app",
        vector_db_docs=vector_db.collection.count(),
        generation_service_url=GENERATION_SERVICE_URL,
        uptime_seconds=round(time.time() - startup_time, 2)
    )


@app.get("/ready")
async def readiness_check():
    """
    Readiness probe - checks if we can reach generation service.
    """
    try:
        # Quick check to generation service
        response = requests.get(
            GENERATION_SERVICE_URL.replace("/api/generate", "/api/tags"),
            timeout=5
        )
        if response.status_code == 200:
            return {"status": "ready", "generation_service": "reachable"}
    except Exception as e:
        logger.warning(f"Generation service not reachable: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Generation service not ready: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest):
    """
    Execute the full RAG pipeline:
    1. Retrieval (CPU-bound) - Embed query, search vector DB
    2. Generation (GPU-bound) - Send to generation-service for LLM inference
    
    This endpoint is the main workload for scheduling experiments.
    """
    global query_counter
    query_counter += 1
    current_query_id = query_counter
    
    total_start = time.time()
    timestamp = datetime.utcnow().isoformat()
    
    logger.info(f"[Query {current_query_id}] Starting RAG pipeline for: {request.prompt[:50]}...")
    
    try:
        # ================================================================
        # STAGE 1: RETRIEVAL (CPU-BOUND)
        # ================================================================
        retrieval_start = time.time()
        
        context_docs = vector_db.retrieve(request.prompt, top_k=request.top_k)
        
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        logger.info(f"[Query {current_query_id}] Retrieval completed in {retrieval_time_ms:.2f}ms, found {len(context_docs)} docs")
        
        # ================================================================
        # STAGE 2: GENERATION (GPU-BOUND VIA NETWORK)
        # ================================================================
        generation_start = time.time()
        
        # Construct augmented prompt with context
        if request.include_context and context_docs:
            context_str = "\n".join(f"- {doc}" for doc in context_docs)
            augmented_prompt = f"""Use the following context to answer the question.

Context:
{context_str}

Question: {request.prompt}

Answer:"""
        else:
            augmented_prompt = request.prompt
        
        # Call generation service (Ollama)
        payload = {
            "model": "gemma:2b",
            "prompt": augmented_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 256,  # Limit output tokens
                "top_p": 0.9
            }
        }
        
        logger.info(f"[Query {current_query_id}] Sending to generation service...")
        
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
            logger.error(f"[Query {current_query_id}] Generation timeout after {GENERATION_TIMEOUT}s")
            raise HTTPException(
                status_code=504,
                detail=f"Generation service timeout after {GENERATION_TIMEOUT}s"
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[Query {current_query_id}] Cannot connect to generation service: {e}")
            raise HTTPException(
                status_code=503,
                detail="Generation service unavailable"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"[Query {current_query_id}] Generation request failed: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Generation service error: {str(e)}"
            )
        
        generation_time_ms = (time.time() - generation_start) * 1000
        logger.info(f"[Query {current_query_id}] Generation completed in {generation_time_ms:.2f}ms")
        
        # ================================================================
        # RESPONSE ASSEMBLY
        # ================================================================
        total_time_ms = (time.time() - total_start) * 1000
        
        logger.info(f"[Query {current_query_id}] Total pipeline: {total_time_ms:.2f}ms "
                   f"(retrieval: {retrieval_time_ms:.2f}ms, generation: {generation_time_ms:.2f}ms)")
        
        # Record metrics
        metrics_entry = {
            "query_id": current_query_id,
            "timestamp": timestamp,
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "generation_time_ms": round(generation_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "success": True
        }
        metrics_history.append(metrics_entry)
        
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
        logger.error(f"[Query {current_query_id}] Pipeline error: {e}", exc_info=True)
        
        # Record failed query
        total_time_ms = (time.time() - total_start) * 1000
        metrics_entry = {
            "query_id": current_query_id,
            "timestamp": timestamp,
            "retrieval_time_ms": 0,
            "generation_time_ms": 0,
            "total_time_ms": round(total_time_ms, 2),
            "success": False,
            "error": str(e)
        }
        metrics_history.append(metrics_entry)
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Return latency metrics history.
    
    This endpoint is consumed by the Bandit scheduler for reward calculation.
    The scheduler polls this to get feedback on scheduling decisions.
    """
    history_list = list(metrics_history)
    
    # Calculate statistics if we have data
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
    """Clear metrics history. Useful between experiment runs."""
    global query_counter
    metrics_history.clear()
    query_counter = 0
    return {"status": "cleared", "message": "Metrics history reset"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "RAG Application",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Execute RAG pipeline",
            "GET /metrics": "Get latency metrics",
            "GET /health": "Health check",
            "GET /ready": "Readiness check"
        }
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
