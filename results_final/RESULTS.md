# Experimental Results — L7 Smart Gateway

## Setup

| Parameter | Value |
|-----------|-------|
| Cluster | 2-node k3s: GPU node (RTX 3070) + CPU node (i5-9500) |
| Model | gemma:2b via Ollama |
| Queries per run | 400 |
| Arrival rate | Poisson λ=1.0 req/s → λ=12.0 at midpoint (dynamic overload ramp) |
| Concurrency limit | 32 |
| Generation timeout (SLA) | 15 s |
| Runs | 10 (4,000 queries per mode across all runs) |

---

## Routing Modes

| Mode | Description |
|------|-------------|
| `baseline` | Always route to GPU |
| `least_in_flight` | Route to node with fewer in-flight requests |
| `static` | Random Forest classifier (3 features: prompt length, GPU/CPU inflight) |
| `bandit_plain` | Thompson Sampling only |
| `bandit_regime` | Thompson Sampling + regime detection + posterior soft-reset |
| `adaptive` | Thompson Sampling + regime detection + 3-state circuit breaker |

---

## Aggregated Results (10 runs, 4,000 queries per mode)

| Mode | Success Rate | Mean (ms) | Normal phase (ms) | Overload phase (ms) | P95 (ms) | P99 (ms) | % GPU |
|------|-------------|-----------|-------------------|---------------------|----------|----------|-------|
| Baseline | **89.9%** | 6,711 | 1,591 | 13,123 | 15,499 | 17,401 | 100% |
| Bandit (plain) | 89.2% | 6,828 | 1,703 | 13,160 | 15,601 | 17,485 | 99% |
| Bandit (regime) | 87.5% | 5,918 | 1,558 | 11,506 | 15,493 | 17,303 | 99% |
| **Adaptive** | 85.7% | **5,567** | 1,633 | **10,872** | 15,402 | 17,174 | 99% |
| Static ML | 80.7% | 3,545 | 1,945 | 6,083 | 8,543 | 14,385 | 97% |
| Least In-Flight | 80.3% | 4,754 | 2,070 | 8,362 | 10,692 | 12,474 | 96% |

Latency values in this table are computed over successful requests only. For a failure-aware
summary, use `Eff. lat (ms)` in `aggregated_comparison_table.csv`.

---

## Key Statistical Findings

| Comparison | Phase | MWU p-value | Cohen's d | Interpretation |
|------------|-------|--------------|-----------|----------------|
| Baseline vs Bandit (plain) | Full | p = 0.098 | d = −0.019 (negligible) | TS matches GPU oracle with zero configuration |
| Bandit (plain) vs Adaptive | Overload | p < 0.001 | d = 0.604 (medium) | Adaptive 17.4% lower overload latency |
| Bandit (regime) vs Adaptive | Full | p = 0.151 | d = 0.063 (negligible) | CB adds negligible improvement over regime detection alone |
| Baseline vs Static ML | Full | p < 0.001 | d = 0.655 (medium) | Static 47% lower mean latency but −9.2pp success rate |

These table values are the raw Mann-Whitney U p-values pulled from
`aggregated_statistical_tests.json`. Bonferroni-corrected values are also included there.
With 6 routing modes, the familywise correction spans 15 pairwise comparisons.

---

## Narrative

**bandit_plain is the success-rate champion.** Thompson Sampling alone matches the always-GPU oracle (p = 0.098, negligible effect size), requiring no prior knowledge of node capabilities. It outperforms the trained static classifier by 8.5 percentage points in success rate.

**adaptive wins on overload latency.** During the high-load phase (the second half of each
400-query run, pooled across runs), the 3-state circuit breaker detects GPU degradation and
redirects traffic, reducing mean overload latency by 17.4% vs bandit_plain (10.9 s vs 13.1 s,
medium effect). The cost is −3.5pp success rate, reflecting occasional misroutes to the CPU
node under sustained flood.

**Static ML and Least-In-Flight are negative results.** Both blindly route ~3–4% of requests to the CPU node during overload. At λ=12 the CPU queue saturates (throughput ≈ 1 req/15 s), turning each CPU-routed request into a timeout. This produces −9pp success rate vs baseline despite lower mean latency (fewer slow GPU requests means the mean is pulled down by the timeouts being excluded as failures).

---

## Figures

| File | Description |
|------|-------------|
| `latency_comparison.png` | Mean latency per mode with 95% CI |
| `boxplot_latency_normal.png` | Latency distribution — normal phase |
| `boxplot_latency_overload.png` | Latency distribution — overload phase |
| `percentile_comparison.png` | P50 / P95 / P99 per mode |
| `success_rate_over_time_avg.png` | Success rate over query index (averaged across 10 runs) |
| `latency_over_time_avg.png` | Mean latency over query index (averaged across 10 runs) |
| `routing_distribution.png` | GPU vs CPU routing split per mode |
| `token_throughput.png` | Token throughput (tok/s) per mode |
| `recovery_chart.png` | Post-overload recovery latency |
| `pre_post_success_comparison.png` | Success rate: normal vs overload phase |
| `per_run_per_run_overload_success.png` | Per-run overload success rate (variance across 10 runs) |
| `failure_breakdown.png` | Failure type breakdown (timeout vs connection error) |
| `aggregated_comparison_table.csv` | Full numeric results table |
| `aggregated_statistical_tests.json` | All pairwise statistical test results |
