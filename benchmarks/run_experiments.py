#!/usr/bin/env python3
import argparse
import asyncio
import atexit
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

DEFAULT_QUERIES     = 100
DEFAULT_RATE        = 1.0
DEFAULT_RAMP_RATE   = 4.0
DEFAULT_CONCURRENCY = 8
DEFAULT_TIMEOUT     = 180
PORT_FORWARD_LOCAL_PORT = 8080

_PROMPTS_FILE = Path(__file__).parent / "prompts.json"


def load_prompts() -> list[str]:
    if _PROMPTS_FILE.exists():
        with open(_PROMPTS_FILE) as f:
            prompts = json.load(f)
        if prompts:
            return prompts
    return [
        "What is Kubernetes and how does it work?",
        "Explain Thompson Sampling for multi-armed bandits.",
        "What is Retrieval-Augmented Generation?",
        "How does GPU acceleration improve LLM inference?",
        "Describe the exploration-exploitation tradeoff.",
    ]


class PortForwardManager:
    def __init__(
        self,
        service: str = "svc/smart-gateway-service",
        local_port: int = PORT_FORWARD_LOCAL_PORT,
        remote_port: int = 8000,
        namespace: str = "default",
    ):
        self.service = service
        self.local_port = local_port
        self.remote_port = remote_port
        self.namespace = namespace
        self.process = None
        self._restart_count = 0

    def start(self):
        self.stop()
        cmd = [
            "kubectl", "port-forward",
            self.service,
            f"{self.local_port}:{self.remote_port}",
            "-n", self.namespace,
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        if self.process.poll() is not None:
            raise RuntimeError(
                f"port-forward exited immediately (code {self.process.returncode})"
            )
        action = "Started" if self._restart_count == 0 else "Restarted"
        print(f"  [{action}] kubectl port-forward {self.service} "
              f"{self.local_port}:{self.remote_port}")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure_alive(self):
        if not self.is_alive():
            self._restart_count += 1
            print(f"\n  [!] port-forward died — restarting (#{self._restart_count})...")
            self.start()

    def get_url(self) -> str:
        return f"http://localhost:{self.local_port}"


def get_gateway_url() -> str:
    return f"http://localhost:{PORT_FORWARD_LOCAL_PORT}"


async def wait_for_service(url: str, timeout: int = 300) -> bool:
    print(f"Waiting for gateway at {url}...")
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                r = await client.get(f"{url}/health", timeout=5.0)
                if r.status_code == 200:
                    print("[OK] Gateway is ready!")
                    return True
            except Exception:
                pass
            await asyncio.sleep(5)
            print(f"  Waiting... ({int(time.time()-start)}s / {timeout}s)")
    print("[FAIL] Gateway did not become ready in time")
    return False


async def clear_metrics(url: str):
    async with httpx.AsyncClient() as client:
        try:
            await client.delete(f"{url}/metrics", timeout=10.0)
            print("[OK] Metrics cleared")
        except Exception as e:
            print(f"Warning: could not clear metrics: {e}")


N_WARMUP = 3
_WARMUP_PROMPTS = [
    "What is Kubernetes?",
    "Explain GPU acceleration for neural networks.",
    "What is retrieval-augmented generation?",
]

async def warmup(url: str):
    print(f"[warmup] sending {N_WARMUP} warmup queries (results discarded)...")
    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(_WARMUP_PROMPTS[:N_WARMUP]):
            try:
                r = await client.post(
                    f"{url}/query",
                    json={"prompt": prompt, "include_context": False, "top_k": 1},
                    timeout=httpx.Timeout(120.0),
                )
                status = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                status = str(e)[:60]
            print(f"  [warmup] q{i + 1}: {status}")
    print("[warmup] done")


async def send_query(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    pf_manager: PortForwardManager = None,
) -> dict:
    loop = asyncio.get_running_loop()

    for attempt in range(max_retries):
        if pf_manager:
            await loop.run_in_executor(None, pf_manager.ensure_alive)

        start_time = time.time()
        try:
            response = await client.post(
                f"{url}/query",
                json={"prompt": prompt, "include_context": True, "top_k": 3},
                timeout=httpx.Timeout(timeout),
            )
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "query_id": data.get("query_id"),
                    "prompt": prompt,
                    "response_length": len(data.get("response", "")),
                    "retrieval_time_ms": data.get("retrieval_time_ms"),
                    "generation_time_ms": data.get("generation_time_ms"),
                    "total_time_ms": data.get("total_time_ms"),
                    "routed_to": data.get("routed_to"),
                    "tokens_generated": data.get("tokens_generated"),
                    "tokens_per_sec": data.get("tokens_per_sec"),
                    "measured_time_ms": elapsed,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                return {
                    "success": False,
                    "prompt": prompt,
                    "error": f"HTTP {response.status_code}",
                    "measured_time_ms": elapsed,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except httpx.TimeoutException:
            elapsed = (time.time() - start_time) * 1000
            return {
                "success": False,
                "prompt": prompt,
                "error": "Timeout",
                "measured_time_ms": elapsed,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except httpx.ConnectError:
            elapsed = (time.time() - start_time) * 1000
            if attempt < max_retries - 1:
                if pf_manager:
                    await loop.run_in_executor(None, pf_manager.start)
                else:
                    await asyncio.sleep(2)
                continue
            return {
                "success": False,
                "prompt": prompt,
                "error": "Connection failed after retries",
                "measured_time_ms": elapsed,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return {
                "success": False,
                "prompt": prompt,
                "error": str(e),
                "measured_time_ms": elapsed,
                "timestamp": datetime.utcnow().isoformat(),
            }


async def run_experiment(
    url: str,
    n_queries: int,
    config_name: str,
    output_dir: Path,
    rate: float = DEFAULT_RATE,
    ramp_rate: float | None = DEFAULT_RAMP_RATE,
    concurrency: int = DEFAULT_CONCURRENCY,
    pf_manager: PortForwardManager = None,
) -> dict:
    prompts = load_prompts()

    overload_index = n_queries // 2
    ramp_desc = f"Rate ramp {rate:.1f} → {ramp_rate:.1f} req/s at Q{overload_index+1}" if ramp_rate else "No ramp"
    print()
    print(f"Experiment: {config_name}")
    print(f"Queries: {n_queries} | Rate: {rate:.1f} req/s | Concurrency: {concurrency}")
    print(f"{ramp_desc}")
    print()

    sem = asyncio.Semaphore(concurrency)
    results_list: list = [None] * n_queries

    async def dispatch_one(i: int, qnum: int, prompt: str, client: httpx.AsyncClient):
        async with sem:
            result = await send_query(client, url, prompt, pf_manager=pf_manager)
            result["query_number"] = qnum
            result["config"] = config_name
            results_list[i] = result

    async with httpx.AsyncClient(timeout=httpx.Timeout(DEFAULT_TIMEOUT)) as client:
        tasks = []
        pbar = tqdm(total=n_queries, desc="Dispatching queries")

        for i in range(n_queries):
            qnum = i + 1
            prompt = prompts[i % len(prompts)]
            task = asyncio.create_task(dispatch_one(i, qnum, prompt, client))
            tasks.append(task)
            pbar.update(1)

            if i == overload_index and ramp_rate is not None:
                print(f"\n[!] RATE RAMP: {rate:.1f} → {ramp_rate:.1f} req/s (query {i+1})")
                rate = ramp_rate

            if i < n_queries - 1:
                inter_arrival = np.random.exponential(1.0 / rate)
                await asyncio.sleep(inter_arrival)

        pbar.close()
        print(f"All {n_queries} queries dispatched — waiting for in-flight requests...")
        await asyncio.gather(*tasks)

    results = results_list
    successful = [r for r in results if r and r["success"]]
    success_count = len(successful)

    if successful:
        latencies = [r["total_time_ms"] for r in successful]
        std = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0
        stats = {
            "config": config_name,
            "total_queries": n_queries,
            "overload_query": overload_index + 1,
            "successful_queries": success_count,
            "success_rate": success_count / n_queries,
            "mean_latency_ms": float(np.mean(latencies)),
            "min_latency_ms": float(min(latencies)),
            "max_latency_ms": float(max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "std_latency_ms": std,
            "timestamp": datetime.utcnow().isoformat(),
        }
        token_rates = [r["tokens_per_sec"] for r in successful if r.get("tokens_per_sec")]
        if token_rates:
            stats["mean_tokens_per_sec"]   = round(float(np.mean(token_rates)),   2)
            stats["median_tokens_per_sec"] = round(float(np.median(token_rates)), 2)
    else:
        stats = {
            "config": config_name,
            "total_queries": n_queries,
            "successful_queries": 0,
            "success_rate": 0,
            "error": "All queries failed",
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{config_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Detailed results: {results_file}")

    summary_file = output_dir / f"{config_name}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Summary: {summary_file}")

    print()
    print("Experiment summary")
    print(f"Config:       {config_name}")
    print(f"Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    if stats.get("mean_latency_ms"):
        print(f"Mean latency: {stats['mean_latency_ms']:.2f}ms")
        print(f"P50 latency:  {stats['p50_latency_ms']:.2f}ms")
        print(f"P95 latency:  {stats['p95_latency_ms']:.2f}ms")
        print(f"P99 latency:  {stats['p99_latency_ms']:.2f}ms")
    if stats.get("mean_tokens_per_sec"):
        print(f"Mean tok/s:   {stats['mean_tokens_per_sec']:.2f}")
    print()

    return stats


async def async_main():
    parser = argparse.ArgumentParser(
        description="Run routing experiments on the L7 Smart Gateway"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Experiment name for output files (e.g., baseline, bandit, static)"
    )
    parser.add_argument(
        "--queries", type=int, default=DEFAULT_QUERIES,
        help=f"Number of queries (default: {DEFAULT_QUERIES})"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RATE,
        help=f"Mean query arrival rate req/s — normal phase (default: {DEFAULT_RATE})"
    )
    parser.add_argument(
        "--ramp-rate", type=float, default=DEFAULT_RAMP_RATE,
        help=f"Arrival rate after overload ramp at midpoint (default: {DEFAULT_RAMP_RATE})"
    )
    parser.add_argument(
        "--no-ramp", action="store_true",
        help="Disable rate ramp (constant rate throughout)"
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="Mean inter-arrival delay in seconds — alias for 1/--rate"
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent in-flight requests (default: {DEFAULT_CONCURRENCY})"
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="Gateway URL (defaults to localhost port-forward)"
    )
    parser.add_argument(
        "--skip-wait", action="store_true",
        help="Skip waiting for service to be ready"
    )
    parser.add_argument(
        "--port-forward", action="store_true",
        help="Manage kubectl port-forward automatically (recommended on Windows/WSL2)"
    )

    args = parser.parse_args()
    rate = 1.0 / args.delay if args.delay is not None else args.rate
    ramp_rate = None if args.no_ramp else args.ramp_rate

    pf_manager = None
    if args.port_forward:
        pf_manager = PortForwardManager()
        pf_manager.start()
        atexit.register(pf_manager.stop)
        url = pf_manager.get_url()
    else:
        url = args.url or get_gateway_url()

    print(f"Gateway URL: {url}")

    if not args.skip_wait:
        if not await wait_for_service(url):
            print("Aborting — gateway not ready")
            if pf_manager:
                pf_manager.stop()
            sys.exit(1)

    await warmup(url)
    await clear_metrics(url)

    stats = await run_experiment(
        url=url,
        n_queries=args.queries,
        config_name=args.config,
        output_dir=Path(args.output),
        rate=rate,
        ramp_rate=ramp_rate,
        concurrency=args.concurrency,
        pf_manager=pf_manager,
    )

    if pf_manager:
        pf_manager.stop()

    print("Experiment complete!")
    return 0 if stats.get("success_rate", 0) > 0.5 else 1


def main():
    sys.exit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
