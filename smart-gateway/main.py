# smart-gateway/main.py
# L7 Smart Gateway — replaces the rag-app + custom-scheduler pair.
#
# Every incoming /query request goes through three stages:
#   1. Retrieval  — vector similarity search against the in-memory ChromaDB store.
#   2. Routing    — select the target Ollama node (0=GPU, 1=CPU) based on
#                   ROUTING_MODE: "baseline" (always GPU), "bandit" (Thompson
#                   Sampling), "static" (pre-trained Random Forest), or
#                   "adaptive" (Thompson Sampling + circuit-breaker on regime change).
#   3. Generation — proxy the augmented prompt to the chosen Ollama endpoint.
#
# In-flight request counts are tracked with asyncio counters protected by a
# single Lock. The gateway MUST run as exactly 1 replica so these counts are
# globally accurate across all concurrent requests.
#
# Bandit posteriors are updated inline after every request (including failures)
# so the gateway reacts to latency degradation within the same experiment run.

import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vectordb import VectorDB
from algorithms.bandit import ThompsonSamplingBandit
from algorithms.static_ml import StaticMLRouter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GPU_OLLAMA_URL      = os.getenv("GPU_OLLAMA_URL",    "http://ollama-gpu-service:11434")
CPU_OLLAMA_URL      = os.getenv("CPU_OLLAMA_URL",    "http://ollama-cpu-service:11434")
MODEL_NAME          = os.getenv("MODEL_NAME",        "gemma:2b")
ROUTING_MODE        = os.getenv("ROUTING_MODE",      "bandit")   # bandit | static | baseline | adaptive
GENERATION_TIMEOUT  = int(os.getenv("GENERATION_TIMEOUT", "30"))
TARGET_LATENCY_MS   = float(os.getenv("TARGET_LATENCY_MS", "5000"))
BANDIT_WINDOW_SIZE  = int(os.getenv("BANDIT_WINDOW_SIZE", "5"))
STATIC_MODEL_PATH   = os.getenv("STATIC_MODEL_PATH", "/app/model/gateway_model.pkl")
METRICS_HISTORY_SIZE = int(os.getenv("METRICS_HISTORY_SIZE", "500"))

_NODE_URLS  = {0: GPU_OLLAMA_URL, 1: CPU_OLLAMA_URL}
_NODE_NAMES = {0: "gpu",          1: "cpu"}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (all writes protected by the appropriate lock)
# ─────────────────────────────────────────────────────────────────────────────

# In-flight request counts per node — must only be modified while holding _inflight_lock
_inflight_lock = asyncio.Lock()
inflight: dict[int, int] = {0: 0, 1: 0}

metrics_history: deque = deque(maxlen=METRICS_HISTORY_SIZE)
_metrics_lock = asyncio.Lock()
_query_counter = 0
_startup_time = time.time()

# Initialised in lifespan
vector_db:     Optional[VectorDB]             = None
bandit:        Optional[ThompsonSamplingBandit] = None
static_router: Optional[StaticMLRouter]       = None
http_client:   Optional[httpx.AsyncClient]    = None


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, bandit, static_router, http_client

    logger.info(f"starting smart-gateway | mode={ROUTING_MODE} | gpu={GPU_OLLAMA_URL} | cpu={CPU_OLLAMA_URL}")

    vector_db = VectorDB()

    # Always initialise the bandit so /bandit/stats is available regardless of mode.
    # is_adaptive=True adds the circuit-breaker layer for the "adaptive" routing mode.
    bandit = ThompsonSamplingBandit(
        target_latency_ms=TARGET_LATENCY_MS,
        k=1.0,
        window_size=BANDIT_WINDOW_SIZE,
        reset_threshold=0.4,
        is_adaptive=(ROUTING_MODE == "adaptive"),
    )

    if ROUTING_MODE == "static":
        static_router = StaticMLRouter(STATIC_MODEL_PATH)
        logger.info(f"static ml model loaded from {STATIC_MODEL_PATH}")

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(GENERATION_TIMEOUT),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )

    logger.info("smart-gateway ready")
    yield

    await http_client.aclose()


app = FastAPI(title="Smart Gateway", version="1.0.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

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
    routed_to: str
    timestamp: str
    tokens_generated: Optional[int] = None
    tokens_per_sec: Optional[float] = None


class MetricsResponse(BaseModel):
    total_queries: int
    history: list[dict]
    avg_total_ms: Optional[float]
    p50_total_ms: Optional[float]
    p95_total_ms: Optional[float]
    p99_total_ms: Optional[float]
    gpu_in_flight: int
    cpu_in_flight: int


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING HELPERS  (called inside _inflight_lock)
# ─────────────────────────────────────────────────────────────────────────────

def _select_node(augmented_prompt: str) -> int:
    """Return target node id (0=GPU, 1=CPU). Called while holding _inflight_lock."""
    if ROUTING_MODE == "baseline":
        return 0

    if ROUTING_MODE == "static":
        return static_router.predict(augmented_prompt, inflight[0], inflight[1])

    if ROUTING_MODE == "adaptive":
        return bandit.select_arm()

    # bandit (default)
    return bandit.select_arm()


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global _query_counter
    _query_counter += 1
    qid = _query_counter
    total_start = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(f"[q{qid}] {request.prompt[:60]}...")

    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    retrieval_start = time.time()
    loop = asyncio.get_running_loop()
    context_docs = await loop.run_in_executor(
        None, lambda: vector_db.retrieve(request.prompt, top_k=request.top_k)
    )
    retrieval_time_ms = (time.time() - retrieval_start) * 1000

    # Build augmented prompt
    if request.include_context and context_docs:
        context_str = "\n".join(f"- {doc}" for doc in context_docs)
        augmented_prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {request.prompt}\n\nAnswer:"
        )
    else:
        augmented_prompt = request.prompt

    # ── Stage 2: Route ────────────────────────────────────────────────────────
    async with _inflight_lock:
        node = _select_node(augmented_prompt)
        inflight[node] += 1

    node_url  = _NODE_URLS[node]
    node_name = _NODE_NAMES[node]
    logger.info(f"[q{qid}] routing to {node_name}")

    # ── Stage 3: Generate ─────────────────────────────────────────────────────
    payload = {
        "model": MODEL_NAME,
        "prompt": augmented_prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 256, "top_p": 0.9},
    }

    request_failed = False
    node_unreachable = False
    error_code: int = 500
    error_msg:  str = ""
    generated_text = ""
    eval_count: Optional[int] = None
    eval_duration_ns: Optional[int] = None
    tokens_per_sec: Optional[float] = None

    generation_start = time.time()
    try:
        response = await http_client.post(f"{node_url}/api/generate", json=payload)
        response.raise_for_status()
        resp_data = response.json()
        generated_text = resp_data.get("response", "")
        eval_count = resp_data.get("eval_count")
        eval_duration_ns = resp_data.get("eval_duration")
        if eval_count and eval_duration_ns and eval_duration_ns > 0:
            tokens_per_sec = round(eval_count / (eval_duration_ns / 1_000_000_000), 2)

    except httpx.TimeoutException:
        request_failed = True
        error_code, error_msg = 504, f"generation timeout ({GENERATION_TIMEOUT}s) on {node_name}"
        logger.error(f"[q{qid}] timeout on {node_name}")

    except httpx.ConnectError as e:
        request_failed = True
        node_unreachable = True
        error_code, error_msg = 503, f"{node_name} ollama unreachable"
        logger.error(f"[q{qid}] connect error on {node_name}: {e}")

    except httpx.RemoteProtocolError as e:
        request_failed = True
        error_code, error_msg = 502, f"{node_name} unexpectedly severed connection"
        logger.error(f"[q{qid}] connection closed on {node_name}: {e}")

    except httpx.ReadError as e:
        request_failed = True
        error_code, error_msg = 502, f"{node_name} connection reset during read"
        logger.error(f"[q{qid}] read error on {node_name}: {e}")

    except httpx.HTTPStatusError as e:
        request_failed = True
        error_code, error_msg = 502, f"{node_name} returned {e.response.status_code}"
        logger.error(f"[q{qid}] http error on {node_name}: {e}")

    finally:
        generation_time_ms = (time.time() - generation_start) * 1000
        async with _inflight_lock:
            inflight[node] = max(0, inflight[node] - 1)
        # Always update bandit — timeouts and errors are penalised with reward 0.
        # unreachable=True triggers immediate arm block in adaptive mode (no
        # need to wait for regime detection when the node is confirmed dead).
        if ROUTING_MODE in ("bandit", "adaptive"):
            bandit.update(
                node, generation_time_ms,
                timed_out=request_failed,
                unreachable=node_unreachable,
            )

    # ── Record metrics and respond ────────────────────────────────────────────
    total_time_ms = (time.time() - total_start) * 1000

    if request_failed:
        async with _metrics_lock:
            metrics_history.append({
                "query_id": qid,
                "timestamp": timestamp,
                "total_time_ms": round(total_time_ms, 2),
                "routed_to": node_name,
                "success": False,
                "error": error_msg,
            })
        raise HTTPException(status_code=error_code, detail=error_msg)

    logger.info(
        f"[q{qid}] done in {total_time_ms:.0f}ms "
        f"(retrieval={retrieval_time_ms:.0f}ms, gen={generation_time_ms:.0f}ms, "
        f"node={node_name}, tok/s={tokens_per_sec})"
    )

    async with _metrics_lock:
        metrics_history.append({
            "query_id": qid,
            "timestamp": timestamp,
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "generation_time_ms": round(generation_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "routed_to": node_name,
            "success": True,
            "tokens_generated": eval_count,
            "tokens_per_sec": tokens_per_sec,
        })

    return QueryResponse(
        query_id=qid,
        response=generated_text,
        context_docs=context_docs,
        retrieval_time_ms=round(retrieval_time_ms, 2),
        generation_time_ms=round(generation_time_ms, 2),
        total_time_ms=round(total_time_ms, 2),
        routed_to=node_name,
        timestamp=timestamp,
        tokens_generated=eval_count,
        tokens_per_sec=tokens_per_sec,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "smart-gateway",
        "routing_mode": ROUTING_MODE,
        "vector_db_docs": vector_db.collection.count() if vector_db else 0,
        "gpu_ollama_url": GPU_OLLAMA_URL,
        "cpu_ollama_url": CPU_OLLAMA_URL,
        "uptime_seconds": round(time.time() - _startup_time, 2),
    }


@app.get("/ready")
async def ready():
    errors = []
    for node_id, url in _NODE_URLS.items():
        try:
            r = await http_client.get(f"{url}/api/tags", timeout=5.0)
            if r.status_code != 200:
                errors.append(f"{_NODE_NAMES[node_id]}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{_NODE_NAMES[node_id]}: {str(e)[:80]}")
    if errors:
        raise HTTPException(status_code=503, detail=f"nodes not ready: {errors}")
    return {"status": "ready"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    async with _metrics_lock:
        history_list = list(metrics_history)

    successful = [m for m in history_list if m.get("success")]
    totals = sorted(m["total_time_ms"] for m in successful)
    n = len(totals)

    async with _inflight_lock:
        gpu_q = inflight[0]
        cpu_q = inflight[1]

    return MetricsResponse(
        total_queries=_query_counter,
        history=history_list,
        avg_total_ms=round(sum(totals) / n, 2) if n else None,
        p50_total_ms=round(totals[int(n * 0.50)], 2) if n else None,
        p95_total_ms=round(totals[min(int(n * 0.95), n - 1)], 2) if n else None,
        p99_total_ms=round(totals[min(int(n * 0.99), n - 1)], 2) if n else None,
        gpu_in_flight=gpu_q,
        cpu_in_flight=cpu_q,
    )


@app.delete("/metrics")
async def clear_metrics():
    global _query_counter
    async with _metrics_lock:
        metrics_history.clear()
        _query_counter = 0
    if bandit is not None:
        bandit.reset()
    return {"status": "cleared"}


@app.get("/bandit/stats")
async def bandit_stats():
    if bandit is None:
        return {"message": "bandit not initialised"}
    return {"routing_mode": ROUTING_MODE, **bandit.get_stats()}


@app.get("/")
async def root():
    return {
        "service": "Smart Gateway",
        "version": "1.0.0",
        "routing_mode": ROUTING_MODE,
        "endpoints": {
            "POST /query":       "submit a RAG query",
            "GET /metrics":      "latency history and current queue depths",
            "DELETE /metrics":   "reset metrics between experiment runs",
            "GET /health":       "health check",
            "GET /ready":        "readiness check (both ollama nodes reachable)",
            "GET /bandit/stats": "bandit posterior state",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
