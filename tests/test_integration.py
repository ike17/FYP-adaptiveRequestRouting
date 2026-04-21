import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smart-gateway"))

import main
from algorithms.static_ml import StaticMLRouter

PKL_PATH = str(ROOT / "smart-gateway" / "model" / "gateway_model.pkl")

QUERY_PAYLOAD = {
    "prompt": "What is Kubernetes scheduling?",
    "include_context": True,
    "top_k": 3,
}


def _mock_ollama_response():
    """Canned httpx.Response shaped like Ollama /api/generate output."""
    mock = MagicMock(spec=httpx.Response)
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "response": "test answer",
        "eval_count": 10,
        "eval_duration": 1_000_000_000,
    }
    return mock


def _make_mock_http_client():
    """Return a MagicMock shaped like httpx.AsyncClient with an async .post."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=_mock_ollama_response())
    return mock_client


# ---------------------------------------------------------------------------
# Session-scoped fixture to initialise shared singletons once per test run.
#
# ASGITransport does NOT trigger the FastAPI lifespan, so vector_db, bandit,
# and http_client remain None after import.  We bootstrap them here so the
# application code can run without a live server.
#
# VectorDB is expensive (loads embeddings) and ChromaDB's default in-memory
# client will raise UniqueConstraintError if create_collection is called twice
# in the same process, so we create it exactly once at session scope.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _init_shared_singletons():
    """Create vector_db and bandit once for the entire test session."""
    from algorithms.bandit import ThompsonSamplingBandit
    from vectordb import VectorDB

    main.vector_db = VectorDB()
    main.bandit = ThompsonSamplingBandit(
        target_latency_ms=5000,
        k=1.0,
        window_size=5,
        reset_threshold=0.4,
        enable_regime_detection=False,
        enable_cb=False,
    )
    yield
    # No teardown needed — in-memory store is discarded with the process.


@pytest.fixture(autouse=True)
def _mock_http_client(monkeypatch):
    """Give every test a fresh mock http_client so tests don't bleed state."""
    monkeypatch.setattr(main, "http_client", _make_mock_http_client())


# ---------------------------------------------------------------------------
# Routing mode parametrization — one /query test per mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [
    "baseline",
    "least_in_flight",
    "bandit_plain",
    "bandit_regime",
    "adaptive",
    "static",
])
async def test_query_all_routing_modes(mode, monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", mode)
    if mode == "static":
        monkeypatch.setattr(main, "static_router", StaticMLRouter(PKL_PATH))

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        main.http_client.post = AsyncMock(return_value=_mock_ollama_response())
        resp = await client.post("/query", json=QUERY_PAYLOAD)

    assert resp.status_code == 200
    data = resp.json()
    assert data["query_id"] >= 1
    assert isinstance(data["response"], str)
    assert data["routed_to"] in ("gpu", "cpu")
    assert data["total_time_ms"] > 0


# ---------------------------------------------------------------------------
# Endpoint smoke tests (run once under baseline mode)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_metrics_endpoint(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        main.http_client.post = AsyncMock(return_value=_mock_ollama_response())
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries" in data
    assert "history" in data


@pytest.mark.asyncio
async def test_bandit_stats_endpoint(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        resp = await client.get("/bandit/stats")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_root_endpoint(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "routing_mode" in resp.json()


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_returns_504(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        main.http_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )
        resp = await client.post("/query", json=QUERY_PAYLOAD)
    assert resp.status_code == 504


@pytest.mark.asyncio
async def test_connect_error_returns_503(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        main.http_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        resp = await client.post("/query", json=QUERY_PAYLOAD)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Metrics reset test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_metrics_resets_query_counter(monkeypatch):
    monkeypatch.setattr(main, "ROUTING_MODE", "baseline")
    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        main.http_client.post = AsyncMock(return_value=_mock_ollama_response())
        await client.post("/query", json=QUERY_PAYLOAD)
        metrics_before = (await client.get("/metrics")).json()
        assert metrics_before["total_queries"] >= 1

        await client.delete("/metrics")
        metrics_after = (await client.get("/metrics")).json()
        assert metrics_after["total_queries"] == 0
