#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURE_DPI  = 300
FIGURE_SIZE = (10, 6)

EXPERIMENT_ORDER = ["baseline", "least_in_flight", "static", "bandit_plain", "bandit_regime", "adaptive"]

COLORS = {
    "baseline":        "#e74c3c",
    "least_in_flight": "#f39c12",
    "static":          "#2ecc71",
    "bandit_plain":    "#3498db",
    "bandit_regime":   "#1abc9c",
    "adaptive":        "#9b59b6",
}

LABELS = {
    "baseline":        "Baseline",
    "least_in_flight": "Least In-Flight",
    "static":          "Static ML",
    "bandit_plain":    "Bandit (plain)",
    "bandit_regime":   "Bandit (regime)",
    "adaptive":        "Adaptive",
}

OVERLOAD_QUERY = 51


def load_results(results_dir: Path) -> tuple[dict, dict]:
    global OVERLOAD_QUERY
    results   = {}
    summaries = {}
    for file in results_dir.glob("*_results.json"):
        config = file.stem.replace("_results", "")
        with open(file) as f:
            results[config] = json.load(f)
    for file in results_dir.glob("*_summary.json"):
        config = file.stem.replace("_summary", "")
        with open(file) as f:
            summaries[config] = json.load(f)
    for s in summaries.values():
        if "overload_query" in s:
            OVERLOAD_QUERY = s["overload_query"]
            break
    return results, summaries


def get_latencies(data: list, phase: str = "full") -> list[float]:
    successful = [r for r in data if r.get("success")]
    if phase == "pre":
        successful = [r for r in successful if r.get("query_number", 0) < OVERLOAD_QUERY]
    elif phase == "post":
        successful = [r for r in successful if r.get("query_number", 0) >= OVERLOAD_QUERY]
    return [r["total_time_ms"] for r in successful]


def bootstrap_ci(data: list, n_iter: int = 5000, ci: float = 0.95) -> tuple:
    if len(data) < 2:
        m = float(np.mean(data)) if data else 0.0
        return m, m
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_iter)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def calculate_effective_latency(results: list) -> float:
    successful = [r for r in results if r.get('success')]
    if not successful:
        return float('inf')
    total_ms = sum(r.get('measured_time_ms', 0) for r in results)
    return total_ms / len(successful)


def create_latency_comparison_chart(summaries: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in summaries]
    if not configs:
        return
    means  = [summaries[c].get("mean_latency_ms", 0) for c in configs]
    stds   = [summaries[c].get("std_latency_ms", 0) for c in configs]
    colors = [COLORS.get(c, "gray") for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x    = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax.set_title('Smart Gateway — Mean End-to-End Latency by Routing Mode', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.0f}±{std:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'latency_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: latency_comparison.png")


def create_percentile_comparison(summaries: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in summaries and summaries[c].get("p50_latency_ms")]
    if not configs:
        return
    p50s = [summaries[c]["p50_latency_ms"] for c in configs]
    p95s = [summaries[c]["p95_latency_ms"] for c in configs]
    p99s = [summaries[c]["p99_latency_ms"] for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x     = np.arange(len(configs))
    width = 0.25
    ax.bar(x - width, p50s, width, label='P50', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x,         p95s, width, label='P95', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, p99s, width, label='P99', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Percentiles by Routing Mode (P50, P95, P99)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'percentile_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: percentile_comparison.png")


def create_latency_over_time(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for config in configs:
        data = results[config]
        successful = [r for r in data if r.get("success")]
        if not successful:
            continue
        queries   = [r["query_number"] for r in successful]
        latencies = [r["total_time_ms"] for r in successful]
        smooth    = pd.Series(latencies).rolling(window=5, min_periods=1).mean().tolist()
        ax.scatter(queries, latencies, color=COLORS.get(config, 'gray'), alpha=0.15, s=10)
        ax.plot(queries, smooth, label=LABELS.get(config, config), color=COLORS.get(config, 'gray'),
                alpha=0.9, linewidth=2)

    ax.axvline(x=OVERLOAD_QUERY, color='red', linestyle='--', alpha=0.7,
               label=f'Overload (Q{OVERLOAD_QUERY})', linewidth=1.5)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Over Time — Routing Behaviour Under Traffic Overload\n'
                 '(dots = raw, lines = 5-query rolling avg)', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path / 'latency_over_time.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: latency_over_time.png")


def create_routing_distribution(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    gpu_counts = []
    cpu_counts = []

    for config in configs:
        data = results[config]
        successful = [r for r in data if r.get("success")]
        gpu_counts.append(sum(1 for r in successful if r.get("routed_to") == "gpu"))
        cpu_counts.append(sum(1 for r in successful if r.get("routed_to") == "cpu"))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x     = np.arange(len(configs))
    width = 0.35

    bars_gpu = ax.bar(x - width / 2, gpu_counts, width, label='GPU',
                      color='#3498db', edgecolor='black', linewidth=0.5)
    bars_cpu = ax.bar(x + width / 2, cpu_counts, width, label='CPU',
                      color='#e67e22', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Successful Requests', fontsize=12)
    ax.set_title('Routing Distribution — GPU vs CPU per Experiment Mode', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.legend()

    for bar in bars_gpu:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    str(int(h)), ha='center', va='bottom', fontsize=9)
    for bar in bars_cpu:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    str(int(h)), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'routing_distribution.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: routing_distribution.png")


def create_success_rate_over_time(results: dict, output_path: Path, window: int = 10):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for config in configs:
        data = sorted(results[config], key=lambda r: r.get('query_number', 0))
        if not data:
            continue
        rates, qnums = [], []
        for i in range(len(data) - window + 1):
            w = data[i:i + window]
            rates.append(sum(1 for r in w if r.get('success')) / window * 100)
            qnums.append(data[i + window - 1].get('query_number', i + window))
        ax.plot(qnums, rates, label=LABELS.get(config, config), color=COLORS.get(config, 'gray'), linewidth=2)

    ax.axvline(x=OVERLOAD_QUERY, color='red', linestyle='--', alpha=0.7,
               label=f'Overload (Q{OVERLOAD_QUERY})', linewidth=1.5)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel(f'Success Rate % (rolling {window}-query window)', fontsize=12)
    ax.set_title('Success Rate Over Time — Overload Resilience\n'
                 '(all queries on x-axis; failed queries count against rate)', fontsize=14)
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate_over_time.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: success_rate_over_time.png")


def create_pre_post_success_comparison(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    pre_rates, post_rates = [], []
    for config in configs:
        data  = results[config]
        pre   = [r for r in data if r.get('query_number', 0) < OVERLOAD_QUERY]
        post  = [r for r in data if r.get('query_number', 0) >= OVERLOAD_QUERY]
        pre_rates.append(sum(1 for r in pre  if r.get('success')) / max(len(pre),  1) * 100)
        post_rates.append(sum(1 for r in post if r.get('success')) / max(len(post), 1) * 100)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x     = np.arange(len(configs))
    width = 0.35
    n_total = max((len(results[c]) for c in configs), default=100)
    ax.bar(x - width / 2, pre_rates,  width, label=f'Normal (Q1-{OVERLOAD_QUERY-1})',
           color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, post_rates, width, label=f'Overload (Q{OVERLOAD_QUERY}-{n_total})',
           color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Normal vs Overload Phase Success Rate', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.set_ylim([0, 110])
    ax.legend()
    for i, (pre, post) in enumerate(zip(pre_rates, post_rates)):
        ax.text(i - width / 2, pre + 1,  f'{pre:.0f}%',  ha='center', va='bottom', fontsize=8)
        ax.text(i + width / 2, post + 1, f'{post:.0f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / 'pre_post_success_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: pre_post_success_comparison.png")


def create_failure_analysis(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    rows = []
    for config in configs:
        failures = [r for r in results[config] if not r.get('success')]
        total    = len(results[config])
        by_type: dict = {}
        for f in failures:
            err = str(f.get('error', 'unknown'))
            if 'timeout' in err.lower() or 'Timeout' in err:
                key = 'Timeout'
            elif 'connection' in err.lower() or 'unreachable' in err.lower():
                key = 'Connection'
            elif err.startswith('HTTP'):
                key = 'HTTP Error'
            else:
                key = err[:25]
            by_type[key] = by_type.get(key, 0) + 1
        rows.append({
            'mode': config,
            'total': total,
            'failures': len(failures),
            'failure_rate_pct': round(len(failures) / max(total, 1) * 100, 1),
            **{f'err_{k}': v for k, v in by_type.items()}
        })

    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(output_path / 'failure_analysis.csv', index=False)
    print("saved: failure_analysis.csv")
    print("\n--- failure breakdown ---")
    print(df.to_string(index=False))


def create_token_throughput_chart(summaries: dict, output_path: Path):
    configs = [
        c for c in EXPERIMENT_ORDER
        if c in summaries and summaries[c].get("mean_tokens_per_sec")
    ]
    if not configs:
        return

    tok_rates = [summaries[c]["mean_tokens_per_sec"] for c in configs]
    colors    = [COLORS.get(c, "gray") for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x    = np.arange(len(configs))
    bars = ax.bar(x, tok_rates, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Mean Tokens / Second', fontsize=12)
    ax.set_title('Token Generation Throughput by Routing Mode\n'
                 '(higher = faster model output, reflects effective node utilisation)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    for bar, rate in zip(bars, tok_rates):
        ax.annotate(f'{rate:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / 'token_throughput.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: token_throughput.png")


def create_recovery_chart(results: dict, output_path: Path, window: int = 5):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    recovery_queries = {}

    for config in configs:
        data      = results[config]
        successful = sorted(
            [r for r in data if r.get("success")],
            key=lambda r: r.get("query_number", 0)
        )
        if not successful:
            continue

        pre_lats  = [r["total_time_ms"] for r in successful if r.get("query_number", 0) < OVERLOAD_QUERY]
        post_lats = [r["total_time_ms"] for r in successful if r.get("query_number", 0) >= OVERLOAD_QUERY]

        if not pre_lats or not post_lats:
            continue

        pre_mean = float(np.mean(pre_lats))

        post_series = pd.Series(post_lats)
        rolling_avg = post_series.rolling(window=window, min_periods=1).mean().tolist()
        x_vals = list(range(len(rolling_avg)))

        color = COLORS.get(config, 'gray')
        lbl = LABELS.get(config, config)
        ax.plot(x_vals, rolling_avg, label=lbl, color=color, linewidth=2)
        ax.axhline(y=pre_mean, color=color, linestyle=':', alpha=0.5,
                   label=f'{lbl} normal mean ({pre_mean:.0f}ms)')

        threshold = pre_mean * 1.2
        for i, v in enumerate(rolling_avg):
            if v <= threshold:
                recovery_queries[config] = i
                break

    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f'Queries after overload (query {OVERLOAD_QUERY})', fontsize=12)
    ax.set_ylabel('Latency ms (5-query rolling avg)', fontsize=12)
    ax.set_title('Post-overload Latency Recovery\n'
                 '(x=0 is first overload query; dotted lines = normal-phase mean)', fontsize=14)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / 'recovery_chart.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: recovery_chart.png")

    if recovery_queries:
        print("\n--- recovery (first query ≤ normal-phase mean × 1.2) ---")
        for config, q in recovery_queries.items():
            print(f"  {config}: query +{q} after overload")


def cohens_d(a: list, b: list) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled != 0 else 0.0


def run_comparison(label, lat1, lat2, config1, config2, n_comparisons) -> dict | None:
    if not lat1 or not lat2:
        return None

    t_stat,  p_raw  = stats.ttest_ind(lat1, lat2, equal_var=False)
    p_bonf          = min(p_raw * n_comparisons, 1.0)
    u_stat,  p_mwu  = mannwhitneyu(lat1, lat2, alternative='two-sided')
    n1, n2          = len(lat1), len(lat2)
    rank_biserial   = float(2 * (u_stat / (n1 * n2)) - 1)
    p_mwu_bonf      = min(p_mwu * n_comparisons, 1.0)
    ks_stat, p_ks   = ks_2samp(lat1, lat2)
    d               = cohens_d(lat1, lat2)

    effect = (
        "negligible" if abs(d) < 0.2 else
        "small"      if abs(d) < 0.5 else
        "medium"     if abs(d) < 0.8 else
        "large"      if abs(d) < 1.2 else
        "very large"
    )

    ci1_lo, ci1_hi = bootstrap_ci(lat1)
    ci2_lo, ci2_hi = bootstrap_ci(lat2)

    return {
        "phase": label, "config_1": config1, "config_2": config2,
        "n_1": n1, "n_2": n2,
        "mean_1": float(np.mean(lat1)), "mean_2": float(np.mean(lat2)),
        "ci95_1": [ci1_lo, ci1_hi], "ci95_2": [ci2_lo, ci2_hi],
        "std_1": float(np.std(lat1, ddof=1)), "std_2": float(np.std(lat2, ddof=1)),
        "t_statistic": float(t_stat), "p_raw": float(p_raw), "p_bonferroni": float(p_bonf),
        "significant_raw": bool(p_raw < 0.05), "significant_bonferroni": bool(p_bonf < 0.05),
        "mwu_statistic": float(u_stat), "p_mwu": float(p_mwu),
        "p_mwu_bonferroni": float(p_mwu_bonf),
        "significant_mwu": bool(p_mwu < 0.05),
        "significant_mwu_bonferroni": bool(p_mwu_bonf < 0.05),
        "rank_biserial": rank_biserial,
        "ks_statistic": float(ks_stat), "p_ks": float(p_ks),
        "cohens_d": float(d), "effect_size": effect,
        "improvement_pct": float(
            (np.mean(lat1) - np.mean(lat2)) / np.mean(lat1) * 100
        ) if np.mean(lat1) > 0 else 0.0,
    }


def perform_statistical_tests(results: dict, output_path: Path) -> list:
    present = [m for m in EXPERIMENT_ORDER if m in results]
    pairs = [(present[i], present[j]) for i in range(len(present)) for j in range(i+1, len(present))]
    n_comparisons = len(pairs)
    comparisons   = []

    n_total = max((len(results[c]) for c in present), default=100)
    for config1, config2 in pairs:
        for phase in ("full", "pre", "post"):
            lat1 = get_latencies(results[config1], phase)
            lat2 = get_latencies(results[config2], phase)
            phase_label = {
                "full": f"full (q1-{n_total})",
                "pre":  f"normal (q1-{OVERLOAD_QUERY-1})",
                "post": f"overload (q{OVERLOAD_QUERY}-{n_total})",
            }[phase]
            result = run_comparison(phase_label, lat1, lat2, config1, config2, n_comparisons)
            if result:
                comparisons.append(result)

    with open(output_path / 'statistical_tests.json', 'w') as f:
        json.dump(comparisons, f, indent=2)
    print("saved: statistical_tests.json")

    print("\n--- statistical analysis ---")
    print(f"bonferroni correction: alpha/n = 0.05/{n_comparisons} = {0.05/n_comparisons:.4f}")
    print("primary test: Mann-Whitney U (non-parametric, handles right-skewed latency)\n")

    for c in comparisons:
        sig = (
            "*** (bonf)" if c["p_mwu_bonferroni"] < 0.001 else
            "** (bonf)"  if c["p_mwu_bonferroni"] < 0.01  else
            "* (bonf)"   if c["p_mwu_bonferroni"] < 0.05  else
            "* (uncorr)" if c["p_mwu"] < 0.05             else ""
        )
        ci1 = c.get("ci95_1", [0, 0])
        ci2 = c.get("ci95_2", [0, 0])
        print(f"[{c['phase']}] {c['config_1']} vs {c['config_2']}")
        print(f"  {c['mean_1']:.1f}ms [{ci1[0]:.0f}-{ci1[1]:.0f} 95%CI]  vs  "
              f"{c['mean_2']:.1f}ms [{ci2[0]:.0f}-{ci2[1]:.0f} 95%CI]")
        print(f"  MWU: U={c['mwu_statistic']:.0f}, p={c['p_mwu']:.4f}, "
              f"p_bonf={c['p_mwu_bonferroni']:.4f} {sig} | r={c['rank_biserial']:.3f}")
        print(f"  Cohen's d={c['cohens_d']:.3f} ({c['effect_size']}), "
              f"improvement={c['improvement_pct']:+.1f}%")
        print()

    return comparisons


def create_summary_table(summaries: dict, results: dict, output_path: Path) -> pd.DataFrame:
    rows = []
    baseline_mean = summaries.get("baseline", {}).get("mean_latency_ms", 0)

    for config in EXPERIMENT_ORDER:
        if config not in summaries:
            continue
        s = summaries[config]

        pre_mean = post_mean = None
        if config in results:
            pre  = get_latencies(results[config], "pre")
            post = get_latencies(results[config], "post")
            if pre:  pre_mean  = round(float(np.mean(pre)),  1)
            if post: post_mean = round(float(np.mean(post)), 1)

        improvement = 0.0
        if baseline_mean > 0 and s.get("mean_latency_ms"):
            improvement = (baseline_mean - s["mean_latency_ms"]) / baseline_mean * 100

        tail_ratio = None
        if s.get("p99_latency_ms") and s.get("p50_latency_ms") and s["p50_latency_ms"] > 0:
            tail_ratio = round(s["p99_latency_ms"] / s["p50_latency_ms"], 2)

        eff_latency = None
        ci_str      = "n/a"
        if config in results:
            eff_latency = round(calculate_effective_latency(results[config]), 1)
            lats = get_latencies(results[config], "full")
            if lats:
                lo, hi = bootstrap_ci(lats)
                ci_str = f"[{lo:.0f}-{hi:.0f}]"

        gpu_pct = cpu_pct = "n/a"
        if config in results:
            successful = [r for r in results[config] if r.get("success")]
            total_s = len(successful)
            if total_s > 0:
                gpu_n = sum(1 for r in successful if r.get("routed_to") == "gpu")
                gpu_pct = f"{gpu_n/total_s*100:.0f}%"
                cpu_pct = f"{(total_s-gpu_n)/total_s*100:.0f}%"

        tok_s = s.get("mean_tokens_per_sec", None)

        rows.append({
            "Mode":            config,
            "Mean (ms)":       f"{s.get('mean_latency_ms', 0):.1f}",
            "95% CI":          ci_str,
            "Std (ms)":        f"{s.get('std_latency_ms', 0):.1f}",
            "P50 (ms)":        f"{s.get('p50_latency_ms', 0):.1f}",
            "P95 (ms)":        f"{s.get('p95_latency_ms', 0):.1f}",
            "P99 (ms)":        f"{s.get('p99_latency_ms', 0):.1f}",
            "P99/P50":         f"{tail_ratio}" if tail_ratio else "n/a",
            "Eff. lat (ms)":   f"{eff_latency}" if eff_latency else "n/a",
            "Normal mean":   f"{pre_mean}" if pre_mean is not None else "n/a",
            "Overload mean": f"{post_mean}" if post_mean is not None else "n/a",
            "vs baseline":     f"{improvement:+.1f}%",
            "% GPU":           gpu_pct,
            "% CPU":           cpu_pct,
            "Tokens/s":        f"{tok_s:.1f}" if tok_s else "n/a",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path / 'comparison_table.csv', index=False)
    print("saved: comparison_table.csv")
    print("\n--- comparison table ---")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="analyse smart-gateway experiment results")
    parser.add_argument("--input",  type=str, default="results", help="results directory")
    parser.add_argument("--output", type=str, default=None,      help="output directory")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading results from: {input_dir}")
    results, summaries = load_results(input_dir)
    if not results:
        print("no results found")
        sys.exit(1)
    print(f"found: {list(results.keys())}")

    create_latency_comparison_chart(summaries, output_dir)
    create_percentile_comparison(summaries, output_dir)
    create_latency_over_time(results, output_dir)
    create_routing_distribution(results, output_dir)
    create_token_throughput_chart(summaries, output_dir)
    create_recovery_chart(results, output_dir)
    create_success_rate_over_time(results, output_dir)
    create_pre_post_success_comparison(results, output_dir)
    create_failure_analysis(results, output_dir)

    perform_statistical_tests(results, output_dir)
    create_summary_table(summaries, results, output_dir)

    print(f"\ndone — charts and stats in: {output_dir}")


if __name__ == "__main__":
    main()
