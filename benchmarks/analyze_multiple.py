#!/usr/bin/env python3
# benchmarks/analyze_multiple.py
# Aggregates results across multiple runs and produces combined charts and stats.

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

FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)

EXPERIMENT_ORDER = ["baseline", "least_in_flight", "static", "bandit_plain", "bandit_regime", "adaptive"]

LABELS = {
    "baseline":        "Baseline",
    "least_in_flight": "Least In-Flight",
    "static":          "Static ML",
    "bandit_plain":    "Bandit (plain)",
    "bandit_regime":   "Bandit (regime)",
    "adaptive":        "Adaptive",
}

COLORS = {
    "baseline":        "#e74c3c",
    "least_in_flight": "#f39c12",
    "static":          "#2ecc71",
    "bandit_plain":    "#3498db",
    "bandit_regime":   "#1abc9c",
    "adaptive":        "#9b59b6",
}

# Overload query number (1-based). Derived from summary JSON at load time;
# falls back to 51 for legacy results that predate the overload_query field.
OVERLOAD_QUERY = 51

# runs with overall success rate below this are excluded from aggregation
MIN_ACCEPTABLE_SUCCESS_RATE = 0.10


# =============================================================================
# UTILITIES
# =============================================================================

def bootstrap_ci(data: list, n_iter: int = 5000, ci: float = 0.95) -> tuple:
    """Distribution-free bootstrap confidence interval for the mean."""
    if len(data) < 2:
        m = float(np.mean(data)) if data else 0.0
        return m, m
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True))
             for _ in range(n_iter)]
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def calculate_effective_latency(results: list) -> float:
    """Throughput-adjusted latency: total wall-clock time / successful completions."""
    successful = [r for r in results if r.get('success')]
    if not successful:
        return float('inf')
    total_ms = sum(r.get('measured_time_ms', 0) for r in results)
    return total_ms / len(successful)


def get_latencies(data: list, phase: str = "full") -> list[float]:
    successful = [r for r in data if r.get("success")]
    if phase == "pre":
        successful = [r for r in successful if r.get("query_number", 0) < OVERLOAD_QUERY]
    elif phase == "post":
        successful = [r for r in successful if r.get("query_number", 0) >= OVERLOAD_QUERY]
    return [r["total_time_ms"] for r in successful]


# =============================================================================
# DATA LOADING
# =============================================================================

def _discover_run_dirs(base_dir: Path) -> list[Path]:
    """Return sorted list of per-run subdirs.

    Supports both old-style 'run_N' directories and the new timestamped
    'YYYYMMDD_HHMMSS' directories produced by run_multiple.sh.  Dirs are
    sorted lexicographically (timestamps sort chronologically).
    """
    import re
    ts_pattern = re.compile(r"^\d{8}_\d{6}$")
    old_pattern = re.compile(r"^run_\d+$")
    candidates = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        if ts_pattern.match(d.name) or old_pattern.match(d.name):
            candidates.append(d)
    return candidates


def load_aggregated_results(base_dir: Path, num_runs: int) -> dict:
    global OVERLOAD_QUERY
    aggregated = {config: [] for config in EXPERIMENT_ORDER}
    excluded = []

    run_dirs = _discover_run_dirs(base_dir)
    if not run_dirs:
        print(f"Warning: no run directories found in {base_dir}")
        return {}

    for i, run_dir in enumerate(run_dirs, start=1):
        for config in EXPERIMENT_ORDER:
            res_file = run_dir / f"{config}_results.json"
            sum_file = run_dir / f"{config}_summary.json"
            if not res_file.exists():
                continue

            if sum_file.exists():
                with open(sum_file) as f:
                    summary = json.load(f)
                sr = summary.get('success_rate', 1.0)
                if sr < MIN_ACCEPTABLE_SUCCESS_RATE:
                    print(f"WARNING: {run_dir.name} {config} has {sr*100:.0f}% success "
                          f"(< {MIN_ACCEPTABLE_SUCCESS_RATE*100:.0f}%) — EXCLUDED")
                    excluded.append({'run': run_dir.name, 'config': config, 'success_rate': sr})
                    continue
                # Derive overload query from summary (first one wins)
                if "overload_query" in summary and OVERLOAD_QUERY == 51:
                    OVERLOAD_QUERY = summary["overload_query"]

            with open(res_file) as f:
                run_data = json.load(f)
            for r in run_data:
                r["run_id"] = i
            aggregated[config].extend(run_data)

    if excluded:
        print(f"Total excluded: {len(excluded)} run(s)")

    return {k: v for k, v in aggregated.items() if v}


def compute_aggregated_summaries(results: dict) -> dict:
    summaries = {}
    for config, data in results.items():
        successful = [r for r in data if r.get("success")]
        total = len(data)
        n_ok = len(successful)
        if n_ok == 0:
            continue

        latencies = [r["total_time_ms"] for r in successful]
        tok_rates = [r["tokens_per_sec"] for r in successful
                     if r.get("tokens_per_sec") is not None]

        summaries[config] = {
            "config": config,
            "total_queries": total,
            "successful_queries": n_ok,
            "success_rate": n_ok / total if total > 0 else 0,
            "mean_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "std_latency_ms": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
            "effective_latency_ms": round(calculate_effective_latency(data), 1),
            "mean_tokens_per_sec": round(float(np.mean(tok_rates)), 2) if tok_rates else None,
            "median_tokens_per_sec": round(float(np.median(tok_rates)), 2) if tok_rates else None,
        }
    return summaries


# =============================================================================
# CHARTS — latency
# =============================================================================

def create_latency_comparison_chart(summaries: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in summaries]
    if not configs:
        return
    means  = [summaries[c]["mean_latency_ms"] for c in configs]
    stds   = [summaries[c]["std_latency_ms"] for c in configs]
    colors = [COLORS[c] for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax.set_title('Aggregated Mean End-to-End Latency by Routing Mode\n'
                 '(error bars = ±1 std, pooled across all runs)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=11)
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
    configs = [c for c in EXPERIMENT_ORDER
               if c in summaries and summaries[c].get("p50_latency_ms")]
    if not configs:
        return
    p50s = [summaries[c]["p50_latency_ms"] for c in configs]
    p95s = [summaries[c]["p95_latency_ms"] for c in configs]
    p99s = [summaries[c]["p99_latency_ms"] for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(configs))
    width = 0.25
    ax.bar(x - width, p50s, width, label='P50', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x,         p95s, width, label='P95', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, p99s, width, label='P99', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Aggregated Latency Percentiles by Routing Mode (P50 / P95 / P99)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'percentile_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: percentile_comparison.png")


def create_latency_over_time_avg(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for config in configs:
        data = results[config]
        successful = [r for r in data if r.get("success")]
        if not successful:
            continue

        query_stats: dict = {}
        for r in successful:
            q = r["query_number"]
            query_stats.setdefault(q, []).append(r["total_time_ms"])

        queries = sorted(query_stats.keys())
        mean_lats = [np.mean(query_stats[q]) for q in queries]
        smooth = pd.Series(mean_lats).rolling(window=5, min_periods=1).mean().tolist()

        ax.scatter(queries, mean_lats, color=COLORS[config], alpha=0.12, s=8)
        ax.plot(queries, smooth, label=LABELS[config], color=COLORS[config],
                alpha=0.9, linewidth=2)

    ax.axvline(x=OVERLOAD_QUERY, color='red', linestyle='--', alpha=0.7,
               label=f'Overload (Q{OVERLOAD_QUERY})', linewidth=1.5)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Mean Latency (ms) across runs', fontsize=12)
    ax.set_title('Average Latency Over Time — Adaptive Behaviour Under Overload\n'
                 '(dots = per-query mean, lines = 5-query rolling avg)', fontsize=13)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path / 'latency_over_time_avg.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: latency_over_time_avg.png")


def create_latency_boxplots(results: dict, output_path: Path):
    """
    Box plots of per-run post-overload latency distributions for successful queries.
    Shows run-to-run variance — unique to multi-run analysis.
    """
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    for phase, label, fname in [
        ("post", "Overload", "boxplot_latency_overload.png"),
        ("pre",  "Normal",  "boxplot_latency_normal.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        all_data = []
        positions = []
        tick_labels = []
        run_ids_all = sorted(set(
            r["run_id"] for config in configs for r in results[config]
        ))
        n_runs = len(run_ids_all)
        spacing = n_runs + 1

        for ci, config in enumerate(configs):
            data = results[config]
            for ri, run_id in enumerate(run_ids_all):
                run_data = [r for r in data if r.get("run_id") == run_id]
                lats = get_latencies(run_data, phase)
                if lats:
                    pos = ci * spacing + ri
                    all_data.append(lats)
                    positions.append(pos)

        bp = ax.boxplot(all_data, positions=positions, widths=0.7,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=2))

        # Colour by config
        idx = 0
        for ci, config in enumerate(configs):
            run_count = sum(
                1 for ri, run_id in enumerate(run_ids_all)
                if get_latencies([r for r in results[config] if r.get("run_id") == run_id], phase)
            )
            for _ in range(run_count):
                bp['boxes'][idx].set_facecolor(COLORS[config])
                bp['boxes'][idx].set_alpha(0.7)
                idx += 1

        # X-tick group labels
        for ci, config in enumerate(configs):
            center = ci * spacing + (n_runs - 1) / 2
            ax.text(center, ax.get_ylim()[0] - ax.get_ylim()[1] * 0.04,
                    LABELS[config], ha='center', va='top', fontsize=11,
                    fontweight='bold', color=COLORS[config])

        ax.set_xlabel('Routing Mode (each box = one run)', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(f'{label} Latency Distribution per Run\n'
                     '(box = IQR, whiskers = 1.5×IQR, outliers hidden)', fontsize=13)
        ax.set_xticks([])

        # Legend patches
        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=COLORS[c], label=LABELS[c]) for c in configs]
        ax.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_path / fname, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"saved: {fname}")


# =============================================================================
# CHARTS — success rate
# =============================================================================

def create_success_rate_over_time_avg(results: dict, output_path: Path, window: int = 10):
    """
    Rolling success rate averaged across all valid runs.
    """
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for config in configs:
        all_data = results[config]
        run_ids = sorted(set(r.get('run_id', 1) for r in all_data))

        query_success: dict = {}
        for run_id in run_ids:
            run_data = sorted([r for r in all_data if r.get('run_id') == run_id],
                              key=lambda r: r['query_number'])
            for i in range(len(run_data) - window + 1):
                w = run_data[i:i + window]
                q = run_data[i + window - 1]['query_number']
                rate = sum(1 for r in w if r.get('success')) / window * 100
                query_success.setdefault(q, []).append(rate)

        if not query_success:
            continue
        queries = sorted(query_success.keys())
        avg_rates = [np.mean(query_success[q]) for q in queries]
        ax.plot(queries, avg_rates, label=LABELS[config],
                color=COLORS[config], linewidth=2)

    ax.axvline(x=OVERLOAD_QUERY, color='red', linestyle='--', alpha=0.7,
               label=f'Overload (Q{OVERLOAD_QUERY})', linewidth=1.5)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel(f'Avg Success Rate % (rolling {window}-query window)', fontsize=12)
    ax.set_title('Aggregated Success Rate Over Time — Overload Resilience\n'
                 '(averaged across all valid runs)', fontsize=13)
    ax.set_ylim([0, 105])
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate_over_time_avg.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: success_rate_over_time_avg.png")


def create_pre_post_success_comparison(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    pre_rates, post_rates = [], []
    for config in configs:
        data = results[config]
        pre  = [r for r in data if r.get('query_number', 0) < OVERLOAD_QUERY]
        post = [r for r in data if r.get('query_number', 0) >= OVERLOAD_QUERY]
        pre_rates.append(sum(1 for r in pre  if r.get('success')) / max(len(pre),  1) * 100)
        post_rates.append(sum(1 for r in post if r.get('success')) / max(len(post), 1) * 100)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(configs))
    width = 0.35
    ax.bar(x - width / 2, pre_rates,  width, label=f'Normal (Q1–{OVERLOAD_QUERY-1})',
           color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, post_rates, width, label=f'Overload (Q{OVERLOAD_QUERY}–100)',
           color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Aggregated Normal vs Overload Success Rate', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=11)
    ax.set_ylim([0, 110])
    ax.legend()
    for i, (pre, post) in enumerate(zip(pre_rates, post_rates)):
        ax.text(i - width / 2, pre + 1,  f'{pre:.0f}%',  ha='center', va='bottom', fontsize=9)
        ax.text(i + width / 2, post + 1, f'{post:.0f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'pre_post_success_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: pre_post_success_comparison.png")


def create_per_run_success_rates(results: dict, output_path: Path):
    """
    Overload success rate for each individual run, grouped by mode.
    Shows reproducibility — high variance here means the result is noisy.
    """
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    run_ids = sorted(set(
        r.get('run_id', 1) for config in configs for r in results[config]
    ))
    x = np.arange(len(run_ids))
    width = 0.2

    for ci, config in enumerate(configs):
        rates = []
        for run_id in run_ids:
            run_data = [r for r in results[config] if r.get('run_id') == run_id]
            post = [r for r in run_data if r.get('query_number', 0) >= OVERLOAD_QUERY]
            rate = sum(1 for r in post if r.get('success')) / max(len(post), 1) * 100
            rates.append(rate)
        offset = (ci - (len(configs) - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=LABELS[config],
               color=COLORS[config], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('Overload Success Rate (%)', fontsize=12)
    ax.set_title('Overload Success Rate per Run — Reproducibility Check\n'
                 '(consistent bars = stable algorithm behaviour)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Run {i}' for i in run_ids], fontsize=9, rotation=45)
    ax.set_ylim([0, 110])
    ax.legend()
    ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path / 'per_run_per_run_overload_success.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: per_run_per_run_overload_success.png")


# =============================================================================
# CHARTS — routing and throughput
# =============================================================================

def create_routing_distribution(results: dict, output_path: Path):
    """GPU vs CPU routing counts per mode, split normal/overload phase."""
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, phase, title in [
        (axes[0], "pre",  f"Normal (Q1–{OVERLOAD_QUERY-1})"),
        (axes[1], "post", f"Overload (Q{OVERLOAD_QUERY}–100)"),
    ]:
        gpu_counts, cpu_counts = [], []
        for config in configs:
            data = results[config]
            if phase == "pre":
                subset = [r for r in data if r.get('query_number', 0) < OVERLOAD_QUERY]
            else:
                subset = [r for r in data if r.get('query_number', 0) >= OVERLOAD_QUERY]
            ok = [r for r in subset if r.get('success')]
            gpu_counts.append(sum(1 for r in ok if r.get('routed_to') == 'gpu'))
            cpu_counts.append(sum(1 for r in ok if r.get('routed_to') == 'cpu'))

        x = np.arange(len(configs))
        width = 0.35
        b_gpu = ax.bar(x - width / 2, gpu_counts, width, label='GPU',
                       color='#3498db', edgecolor='black', linewidth=0.5)
        b_cpu = ax.bar(x + width / 2, cpu_counts, width, label='CPU',
                       color='#e67e22', edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Routing Mode', fontsize=11)
        ax.set_ylabel('Successful Requests', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in configs], fontsize=10)
        ax.legend()
        for bar in list(b_gpu) + list(b_cpu):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        str(int(h)), ha='center', va='bottom', fontsize=8)

    fig.suptitle('Routing Distribution — GPU vs CPU per Phase (aggregated)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path / 'routing_distribution.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: routing_distribution.png")


def create_token_throughput_chart(results: dict, summaries: dict, output_path: Path):
    """Mean token throughput (tok/s) per mode, split normal/overload phase."""
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(configs))
    width = 0.35

    pre_tok, post_tok = [], []
    for config in configs:
        data = results[config]

        pre_rates = [r["tokens_per_sec"] for r in data
                     if r.get("success") and r.get("tokens_per_sec")
                     and r.get("query_number", 0) < OVERLOAD_QUERY]
        post_rates = [r["tokens_per_sec"] for r in data
                      if r.get("success") and r.get("tokens_per_sec")
                      and r.get("query_number", 0) >= OVERLOAD_QUERY]

        pre_tok.append(float(np.mean(pre_rates)) if pre_rates else 0)
        post_tok.append(float(np.mean(post_rates)) if post_rates else 0)

    b_pre  = ax.bar(x - width / 2, pre_tok,  width,
                    label=f'Normal (Q1–{OVERLOAD_QUERY-1})',
                    color='#2ecc71', edgecolor='black', linewidth=0.5)
    b_post = ax.bar(x + width / 2, post_tok, width,
                    label=f'Overload (Q{OVERLOAD_QUERY}–100)',
                    color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.85)

    for bar in list(b_pre) + list(b_post):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Routing Mode', fontsize=12)
    ax.set_ylabel('Mean Tokens / Second', fontsize=12)
    ax.set_title('Aggregated Token Generation Throughput by Routing Mode\n'
                 '(GPU ~165 tok/s; CPU ~85 tok/s — lower post-overload = more CPU routing)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'token_throughput.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: token_throughput.png")


# =============================================================================
# CHARTS — recovery
# =============================================================================

def create_recovery_chart(results: dict, output_path: Path, window: int = 5):
    """
    Overload latency recovery: per-mode mean rolling latency across runs.
    X-axis = queries after overload (0 = first post-overload query).
    Dotted horizontal = each mode's aggregated normal-phase mean.
    """
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    recovery_queries = {}

    for config in configs:
        data = results[config]
        run_ids = sorted(set(r.get('run_id', 1) for r in data))

        # Collect per-run post-overload successful latencies by relative query index
        rel_lats: dict = {}  # relative_idx → list of latencies across runs
        pre_means = []

        for run_id in run_ids:
            run_data = sorted(
                [r for r in data if r.get('run_id') == run_id],
                key=lambda r: r.get('query_number', 0)
            )
            pre_ok  = [r["total_time_ms"] for r in run_data
                       if r.get("success") and r.get("query_number", 0) < OVERLOAD_QUERY]
            post_ok = [r["total_time_ms"] for r in run_data
                       if r.get("success") and r.get("query_number", 0) >= OVERLOAD_QUERY]
            if pre_ok:
                pre_means.append(float(np.mean(pre_ok)))
            for idx, lat in enumerate(post_ok):
                rel_lats.setdefault(idx, []).append(lat)

        if not rel_lats or not pre_means:
            continue

        pre_mean = float(np.mean(pre_means))
        indices = sorted(rel_lats.keys())
        mean_lats = [float(np.mean(rel_lats[i])) for i in indices]
        smooth = pd.Series(mean_lats).rolling(window=window, min_periods=1).mean().tolist()

        color = COLORS[config]
        ax.plot(indices, smooth, label=LABELS[config], color=color, linewidth=2)
        ax.axhline(y=pre_mean, color=color, linestyle=':', alpha=0.5,
                   label=f'{LABELS[config]} normal-phase ({pre_mean:.0f}ms)')

        threshold = pre_mean * 1.2
        for i, v in enumerate(smooth):
            if v <= threshold:
                recovery_queries[config] = i
                break

    ax.set_xlabel(f'Queries after overload (0 = Q{OVERLOAD_QUERY})', fontsize=12)
    ax.set_ylabel('Mean Latency (ms) — 5-query rolling avg', fontsize=12)
    ax.set_title('Aggregated Overload Latency Recovery\n'
                 '(dotted = normal-phase mean; averaged across all valid runs)', fontsize=13)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path / 'recovery_chart.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("saved: recovery_chart.png")

    if recovery_queries:
        print("\n--- recovery (first query ≤ normal-phase mean × 1.2, avg across runs) ---")
        for config, q in recovery_queries.items():
            print(f"  {LABELS[config]}: query +{q} after overload")


# =============================================================================
# CHARTS — failure analysis
# =============================================================================

def create_failure_analysis(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs:
        return

    rows = []
    for config in configs:
        data = results[config]
        failures = [r for r in data if not r.get('success')]
        total = len(data)
        by_type: dict = {}
        for f in failures:
            err = str(f.get('error', 'unknown'))
            if '504' in err or 'timeout' in err.lower():
                key = 'Timeout (504)'
            elif '503' in err or 'unreachable' in err.lower():
                key = 'Unavailable (503)'
            elif '502' in err or 'connection' in err.lower():
                key = 'Connection (502)'
            else:
                key = err[:25]
            by_type[key] = by_type.get(key, 0) + 1

        rows.append({
            'Mode': LABELS[config],
            'Total': total,
            'Failures': len(failures),
            'Failure Rate': f"{len(failures)/max(total,1)*100:.1f}%",
            **{k: v for k, v in by_type.items()}
        })

    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(output_path / 'failure_analysis.csv', index=False)
    print("saved: failure_analysis.csv")

    # Stacked bar chart of failure types
    error_cols = [c for c in df.columns
                  if c not in ('Mode', 'Total', 'Failures', 'Failure Rate')]
    if error_cols:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        x = np.arange(len(df))
        bottom = np.zeros(len(df))
        err_colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
        for i, col in enumerate(error_cols):
            vals = df[col].values.astype(float)
            ax.bar(x, vals, bottom=bottom, label=col,
                   color=err_colors[i % len(err_colors)],
                   edgecolor='black', linewidth=0.5)
            bottom += vals
        ax.set_xlabel('Routing Mode', fontsize=12)
        ax.set_ylabel('Number of Failures', fontsize=12)
        ax.set_title('Aggregated Failure Breakdown by Error Type', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Mode'].tolist(), fontsize=11)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'failure_breakdown.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("saved: failure_breakdown.png")

    print("\n--- failure breakdown ---")
    print(df.to_string(index=False))


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def cohens_d(a: list, b: list) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = np.std(a, ddof=1)
    s2 = np.std(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled != 0 else 0.0


def run_comparison(label, lat1, lat2, config1, config2, n_comparisons) -> dict | None:
    if not lat1 or not lat2:
        return None
    t_stat, p_raw   = stats.ttest_ind(lat1, lat2, equal_var=False)
    p_bonf           = min(p_raw * n_comparisons, 1.0)
    u_stat, p_mwu    = mannwhitneyu(lat1, lat2, alternative='two-sided')
    n1, n2           = len(lat1), len(lat2)
    rank_biserial    = float(2 * (u_stat / (n1 * n2)) - 1)
    p_mwu_bonf       = min(p_mwu * n_comparisons, 1.0)
    ks_stat, p_ks    = ks_2samp(lat1, lat2)
    d                = cohens_d(lat1, lat2)
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
    # Build pairs dynamically from available modes (in canonical order)
    present = [m for m in EXPERIMENT_ORDER if m in results]
    pairs = [(present[i], present[j]) for i in range(len(present)) for j in range(i+1, len(present))]
    n_comparisons = len(pairs)
    comparisons = []

    # Derive total query count from data for labels
    n_total = max((len(results[c]) for c in present), default=100)
    for config1, config2 in pairs:
        for phase in ("full", "pre", "post"):
            lat1 = get_latencies(results[config1], phase)
            lat2 = get_latencies(results[config2], phase)
            phase_label = {
                "full": f"full (q1-{n_total})",
                "pre":  f"normal-phase (q1-{OVERLOAD_QUERY-1})",
                "post": f"post-overload (q{OVERLOAD_QUERY}-{n_total})",
            }[phase]
            r = run_comparison(phase_label, lat1, lat2, config1, config2, n_comparisons)
            if r:
                comparisons.append(r)

    with open(output_path / 'aggregated_statistical_tests.json', 'w') as f:
        json.dump(comparisons, f, indent=2)
    print("saved: aggregated_statistical_tests.json")

    print("\n--- aggregated statistical analysis ---")
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
        print(f"  {c['mean_1']:.1f}ms [{ci1[0]:.0f}–{ci1[1]:.0f} 95%CI, n={c['n_1']}] vs "
              f"{c['mean_2']:.1f}ms [{ci2[0]:.0f}–{ci2[1]:.0f} 95%CI, n={c['n_2']}]")
        print(f"  MWU: U={c['mwu_statistic']:.0f}, p={c['p_mwu']:.4f}, "
              f"p_bonf={c['p_mwu_bonferroni']:.4f} {sig} | r={c['rank_biserial']:.3f}")
        print(f"  Cohen's d={c['cohens_d']:.3f} ({c['effect_size']}), "
              f"improvement={c['improvement_pct']:+.1f}%")
        print()

    return comparisons


# =============================================================================
# SUMMARY TABLE
# =============================================================================

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
        p99 = s.get("p99_latency_ms")
        p50 = s.get("p50_latency_ms")
        if p99 and p50 and p50 > 0:
            tail_ratio = round(p99 / p50, 2)

        ci_str = "n/a"
        if config in results:
            lats = get_latencies(results[config], "full")
            if lats:
                lo, hi = bootstrap_ci(lats)
                ci_str = f"[{lo:.0f}–{hi:.0f}]"

        gpu_pct = cpu_pct = "n/a"
        if config in results:
            ok = [r for r in results[config] if r.get("success")]
            total_ok = len(ok)
            if total_ok > 0:
                gpu_n   = sum(1 for r in ok if r.get("routed_to") == "gpu")
                gpu_pct = f"{gpu_n/total_ok*100:.0f}%"
                cpu_pct = f"{(total_ok-gpu_n)/total_ok*100:.0f}%"

        tok_s = s.get("mean_tokens_per_sec")
        eff   = s.get("effective_latency_ms")

        rows.append({
            "Mode":             LABELS[config],
            "OK / Total":       f"{s.get('successful_queries',0)} / {s.get('total_queries',0)}",
            "Success Rate":     f"{s.get('success_rate',0)*100:.1f}%",
            "Mean (ms)":        f"{s.get('mean_latency_ms',0):.1f}",
            "95% CI":           ci_str,
            "Std (ms)":         f"{s.get('std_latency_ms',0):.1f}",
            "P50 (ms)":         f"{p50:.1f}" if p50 else "n/a",
            "P95 (ms)":         f"{s.get('p95_latency_ms',0):.1f}",
            "P99 (ms)":         f"{p99:.1f}" if p99 else "n/a",
            "P99/P50":          f"{tail_ratio}" if tail_ratio else "n/a",
            "Eff. lat (ms)":    f"{eff}" if eff else "n/a",
            "Normal (ms)":   f"{pre_mean}" if pre_mean is not None else "n/a",
            "Overload (ms)":  f"{post_mean}" if post_mean is not None else "n/a",
            "vs Baseline":      f"{improvement:+.1f}%",
            "% GPU":            gpu_pct,
            "% CPU":            cpu_pct,
            "Tok/s (mean)":     f"{tok_s:.1f}" if tok_s else "n/a",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path / 'aggregated_comparison_table.csv', index=False)
    print("saved: aggregated_comparison_table.csv")
    print("\n--- aggregated comparison table ---")
    print(df.to_string(index=False))
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse aggregated scheduler experiment results across multiple runs")
    parser.add_argument("--input",  type=str, default="results",
                        help="Base results directory containing run_1, run_2, etc.")
    parser.add_argument("--runs",   type=int, default=10,
                        help="Number of runs to aggregate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for aggregated analysis")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    num_runs   = args.runs
    output_dir = Path(args.output) if args.output else input_dir / "aggregated_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading aggregated results from {num_runs} runs in: {input_dir}")
    print(f"excluding runs with < {MIN_ACCEPTABLE_SUCCESS_RATE*100:.0f}% success rate\n")
    results = load_aggregated_results(input_dir, num_runs)

    if not results:
        print("no results found")
        sys.exit(1)

    print(f"found data for: {list(results.keys())}")
    summaries = compute_aggregated_summaries(results)

    # Latency charts
    create_latency_comparison_chart(summaries, output_dir)
    create_percentile_comparison(summaries, output_dir)
    create_latency_over_time_avg(results, output_dir)
    create_latency_boxplots(results, output_dir)
    create_recovery_chart(results, output_dir)

    # Success rate charts
    create_success_rate_over_time_avg(results, output_dir)
    create_pre_post_success_comparison(results, output_dir)
    create_per_run_success_rates(results, output_dir)

    # Routing and throughput
    create_routing_distribution(results, output_dir)
    create_token_throughput_chart(results, summaries, output_dir)

    # Failure analysis
    create_failure_analysis(results, output_dir)

    # Stats and table
    perform_statistical_tests(results, output_dir)
    create_summary_table(summaries, results, output_dir)

    print(f"\ndone — aggregated results in {output_dir}")


if __name__ == "__main__":
    main()
