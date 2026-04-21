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

## Design Rationale

Thompson Sampling (TS) was chosen over UCB and ε-greedy because it requires no tuning parameter — UCB needs a confidence scalar and ε-greedy needs an exploration rate, both of which require offline calibration. TS naturally balances exploration and exploitation through posterior sampling, adapting the exploration rate implicitly as evidence accumulates.

The reward function uses a pseudo-Bernoulli update: continuous rewards r ∈ [0,1] are fed into α += r, β += (1−r). This is a documented approximation of the Beta-Bernoulli model, appropriate here because the reward is a bounded continuous proxy for a binary latency-SLA outcome (did the request complete within the target?). The alternative — Gaussian Thompson Sampling — requires estimating variance online and is less stable with small window sizes.

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

### Statistical Power Note

The `p = 0.098` result for Baseline vs Bandit (plain) is an **absence-of-evidence** result, not evidence of equivalence. With n=10 runs and an observed effect size of d=−0.019 (negligible), the power to detect a difference of this magnitude is very low at this sample size. The correct interpretation is: *we failed to detect a statistically significant difference at n=10*, not *no difference exists*. A larger replication could confirm or refute equivalence.

Both results marked p < 0.001 (Baseline vs Static ML; Bandit plain vs Adaptive overload) survive Bonferroni correction for 15 pairwise comparisons (corrected α = 0.0033). The Bandit regime vs Adaptive result (p = 0.151) does not survive correction and should be interpreted as inconclusive.

---

## Narrative

**Effective latency is the primary metric.** Successful-only latency (the table above) understates the cost of modes with high failure rates. Effective latency — computed as `(mean_latency × success_rate) + (SLA_timeout × failure_rate)` — is the operationally correct comparison. Full values are in `aggregated_comparison_table.csv`. The narrative below leads with effective latency; success rate is the cost side of each trade-off.

**bandit_plain is the success-rate champion.** Thompson Sampling alone matches the always-GPU oracle (p = 0.098, negligible effect size), requiring no prior knowledge of node capabilities. It outperforms the trained static classifier by 8.5 percentage points in success rate.

**adaptive wins on overload latency.** During the high-load phase (the second half of each 400-query run, pooled across runs), the 3-state circuit breaker detects GPU degradation and redirects traffic, reducing mean overload latency by 17.4% vs bandit_plain (10.9 s vs 13.1 s, medium effect). The cost is −3.5pp success rate, reflecting occasional misroutes to the CPU node under sustained flood.

**Static ML and Least-In-Flight are negative results.** Both blindly route ~3–4% of requests to the CPU node during overload. At λ=12 the CPU queue saturates (throughput ≈ 1 req/15 s), turning each CPU-routed request into a timeout. This produces −9pp success rate vs baseline despite lower mean latency on successful requests (timeouts are excluded as failures, pulling the mean down).

---

## Known Implementation Behaviours

Two implementation behaviours are worth documenting explicitly:

- **Soft-reset oscillation under sustained overload.** When the system is under continuous overload and the circuit breaker is not yet open, `_reset_count` increments at approximately one per `window_size` pulls. This is expected: the regime detector fires repeatedly while the reward window remains degraded. CB intervention breaks the cycle by removing the degraded arm from selection. This is a known design trade-off, not a bug.

- **`_query_counter` asyncio safety.** The query counter is a plain Python integer incremented with `+=` between `await` points. This is safe under CPython's GIL and uvicorn's default single-process single-event-loop model (`workers=1`). Under a multi-worker deployment the counter would require a lock or an external store. The gateway is explicitly documented as requiring exactly one replica.

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

---

## Threats to Validity

- **Single cluster, single workload.** Results reflect one RTX 3070 + i5-9500 pair under a single Poisson arrival pattern (λ=1 → λ=12 ramp). Generalisation to different hardware ratios, multi-node clusters, or non-Poisson workloads is untested.

- **Sequential runs.** The 10 experiment runs are time-ordered on shared hardware. Thermal drift, VRAM fragmentation, and Ollama's internal KV-cache state may correlate consecutive runs, weakening the independent-samples assumption underlying Mann-Whitney U. A randomised run order across days would strengthen the independence claim.

- **Single model.** All experiments use gemma:2b. Models with different generation latency profiles (e.g. larger parameter counts or different quantisations) may shift the crossover point between routing strategies, particularly the overload threshold at which the circuit breaker activates.

- **Hyperparameter sensitivity.** Window size (5), reset threshold (0.4), CB durations (15 s/30 s), and admission rates (25%/50%) were set by reasoning, not tuned on held-out data. No sensitivity sweep was performed. Results may vary under different hyperparameter settings.
