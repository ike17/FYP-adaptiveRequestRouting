#!/usr/bin/env python3
"""
benchmarks/run_experiments.py

Automated experiment runner for scheduler evaluation.
Sends queries to the RAG application and collects latency metrics.

Usage:
    python run_experiments.py --config baseline-optimal --queries 100 --output results/

    # With managed port-forward (recommended on Windows):
    python run_experiments.py --config bandit-adaptive --queries 100 --port-forward
"""

import argparse
import atexit
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default settings
DEFAULT_QUERIES = 100
DEFAULT_DELAY = 0.5  # seconds between queries
DEFAULT_TIMEOUT = 120  # seconds per query
PORT_FORWARD_LOCAL_PORT = 8080

# Sample queries for RAG
SAMPLE_QUERIES = [
    "What is Kubernetes and how does it work?",
    "Explain the difference between pods and containers",
    "How does machine learning improve scheduling?",
    "What is Retrieval-Augmented Generation?",
    "Describe Thompson Sampling algorithm",
    "What are the benefits of GPU acceleration for LLMs?",
    "How do contextual bandits work?",
    "Explain piecewise stationary environments",
    "What is a Random Forest classifier?",
    "How does vector similarity search work?",
    "What are the challenges of heterogeneous clusters?",
    "Explain the exploration-exploitation tradeoff",
    "What is semantic search?",
    "How do transformer models work?",
    "Describe online learning algorithms",
]


# =============================================================================
# PORT-FORWARD MANAGER
# =============================================================================

class PortForwardManager:
    """
    Manages kubectl port-forward lifecycle.

    On Windows/WSL2, port-forward silently dies after ~5-10 minutes.
    This class detects the dead process and restarts it automatically.
    """

    def __init__(self, service: str = "svc/rag-app-service",
                 local_port: int = PORT_FORWARD_LOCAL_PORT,
                 remote_port: int = 8000,
                 namespace: str = "default"):
        self.service = service
        self.local_port = local_port
        self.remote_port = remote_port
        self.namespace = namespace
        self.process = None
        self._restart_count = 0

    def start(self):
        """Start or restart port-forward."""
        self.stop()
        cmd = [
            "kubectl", "port-forward",
            self.service,
            f"{self.local_port}:{self.remote_port}",
            "-n", self.namespace
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give it a moment to bind the port
        time.sleep(2)
        if self.process.poll() is not None:
            raise RuntimeError(
                f"port-forward exited immediately (code {self.process.returncode})"
            )
        action = "Started" if self._restart_count == 0 else "Restarted"
        print(f"  [{action}] kubectl port-forward {self.service} "
              f"{self.local_port}:{self.remote_port}")

    def stop(self):
        """Kill the port-forward process if running."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def is_alive(self) -> bool:
        """Check if the port-forward process is still running."""
        return self.process is not None and self.process.poll() is None

    def ensure_alive(self):
        """Restart port-forward if it has died."""
        if not self.is_alive():
            self._restart_count += 1
            print(f"\n  [!] port-forward died - restarting "
                  f"(restart #{self._restart_count})...")
            self.start()

    def get_url(self) -> str:
        return f"http://localhost:{self.local_port}"


def get_rag_url() -> str:
    return f"http://localhost:{PORT_FORWARD_LOCAL_PORT}"


def wait_for_service(url: str, timeout: int = 300) -> bool:
    """Wait for the RAG service to be ready."""
    print(f"Waiting for service at {url}...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print("[OK] Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(5)
        elapsed = int(time.time() - start)
        print(f"  Waiting... ({elapsed}s / {timeout}s)")
    
    print("[FAIL] Service did not become ready in time")
    return False


def clear_metrics(url: str):
    """Clear metrics history before experiment."""
    try:
        requests.delete(f"{url}/metrics", timeout=10)
        print("[OK] Metrics cleared")
    except Exception as e:
        print(f"Warning: Could not clear metrics: {e}")


def send_query(url: str, prompt: str, timeout: int = DEFAULT_TIMEOUT,
               max_retries: int = 3, pf_manager: PortForwardManager = None) -> dict:
    """
    Send a single query to the RAG application with retry on connection errors.
    If a PortForwardManager is provided, restarts port-forward on connection failure.

    Returns:
        Dict with query results including latency
    """
    for attempt in range(max_retries):
        # Ensure port-forward is alive before each attempt
        if pf_manager:
            pf_manager.ensure_alive()

        start_time = time.time()

        try:
            response = requests.post(
                f"{url}/query",
                json={
                    "prompt": prompt,
                    "include_context": True,
                    "top_k": 3
                },
                timeout=timeout
            )

            elapsed = (time.time() - start_time) * 1000  # ms

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
                    "measured_time_ms": elapsed,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "prompt": prompt,
                    "error": f"HTTP {response.status_code}",
                    "measured_time_ms": elapsed,
                    "timestamp": datetime.utcnow().isoformat()
                }

        except requests.exceptions.Timeout:
            elapsed = (time.time() - start_time) * 1000
            return {
                "success": False,
                "prompt": prompt,
                "error": "Timeout",
                "measured_time_ms": elapsed,
                "timestamp": datetime.utcnow().isoformat()
            }
        except requests.exceptions.ConnectionError:
            elapsed = (time.time() - start_time) * 1000
            if attempt < max_retries - 1:
                if pf_manager:
                    # Force restart - port-forward is dead
                    pf_manager.start()
                    time.sleep(3)
                else:
                    time.sleep(2)
                continue
            return {
                "success": False,
                "prompt": prompt,
                "error": "Connection failed after retries",
                "measured_time_ms": elapsed,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {
                "success": False,
                "prompt": prompt,
                "error": str(e),
                "measured_time_ms": elapsed,
                "timestamp": datetime.utcnow().isoformat()
            }


def run_experiment(
    url: str,
    n_queries: int,
    config_name: str,
    output_dir: Path,
    delay: float = DEFAULT_DELAY,
    pf_manager: PortForwardManager = None
) -> dict:
    """
    Run a full experiment with n_queries.

    Returns:
        Dict with experiment summary
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config_name}")
    print(f"Queries: {n_queries}")
    print(f"Delay: {delay}s")
    if pf_manager:
        print(f"Port-forward: MANAGED (auto-restart on failure)")
    print(f"{'='*60}\n")

    results = []
    success_count = 0

    # Run queries
    for i in tqdm(range(n_queries), desc="Sending queries"):
        # Cycle through sample queries
        prompt = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]

        result = send_query(url, prompt, pf_manager=pf_manager)
        result["query_number"] = i + 1
        result["config"] = config_name
        results.append(result)
        
        if result["success"]:
            success_count += 1
        
        # Delay between queries
        if i < n_queries - 1:
            time.sleep(delay)
    
    # Calculate statistics
    successful = [r for r in results if r["success"]]
    
    if successful:
        latencies = [r["total_time_ms"] for r in successful]
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        mean_latency = sum(latencies) / len(latencies)
        std_latency = (sum((x - mean_latency)**2 for x in latencies) / len(latencies)) ** 0.5 if len(latencies) > 1 else 0

        stats = {
            "config": config_name,
            "total_queries": n_queries,
            "successful_queries": success_count,
            "success_rate": success_count / n_queries,
            "mean_latency_ms": mean_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": sorted_latencies[int(n * 0.50)],
            "p95_latency_ms": sorted_latencies[min(int(n * 0.95), n-1)],
            "p99_latency_ms": sorted_latencies[min(int(n * 0.99), n-1)],
            "std_latency_ms": std_latency,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        stats = {
            "config": config_name,
            "total_queries": n_queries,
            "successful_queries": 0,
            "success_rate": 0,
            "error": "All queries failed"
        }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / f"{config_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Detailed results saved: {results_file}")

    # Save summary
    summary_file = output_dir / f"{config_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Summary saved: {summary_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Config: {config_name}")
    print(f"Success rate: {stats.get('success_rate', 0)*100:.1f}%")
    if stats.get('mean_latency_ms'):
        print(f"Mean latency: {stats['mean_latency_ms']:.2f}ms")
        print(f"P50 latency: {stats['p50_latency_ms']:.2f}ms")
        print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
        print(f"P99 latency: {stats['p99_latency_ms']:.2f}ms")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run scheduling experiments on RAG application"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Experiment configuration name (e.g., baseline-optimal, static-ml)"
    )
    parser.add_argument(
        "--queries", 
        type=int, 
        default=DEFAULT_QUERIES,
        help=f"Number of queries to send (default: {DEFAULT_QUERIES})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=DEFAULT_DELAY,
        help=f"Delay between queries in seconds (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="RAG app URL (defaults to localhost port-forward)"
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip waiting for service to be ready"
    )
    parser.add_argument(
        "--port-forward",
        action="store_true",
        help="Manage kubectl port-forward automatically (recommended on Windows)"
    )

    args = parser.parse_args()

    # Port-forward management
    pf_manager = None
    if args.port_forward:
        pf_manager = PortForwardManager()
        pf_manager.start()
        atexit.register(pf_manager.stop)
        url = pf_manager.get_url()
    else:
        url = args.url or get_rag_url()

    print(f"RAG URL: {url}")

    # Wait for service
    if not args.skip_wait:
        if not wait_for_service(url):
            print("Aborting experiment - service not ready")
            if pf_manager:
                pf_manager.stop()
            sys.exit(1)

    # Clear previous metrics
    clear_metrics(url)

    # Run experiment
    output_dir = Path(args.output)
    stats = run_experiment(
        url=url,
        n_queries=args.queries,
        config_name=args.config,
        output_dir=output_dir,
        delay=args.delay,
        pf_manager=pf_manager
    )

    # Cleanup
    if pf_manager:
        pf_manager.stop()

    print("Experiment complete!")
    return 0 if stats.get("success_rate", 0) > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
