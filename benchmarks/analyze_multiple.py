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

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)

EXPERIMENT_ORDER = ["baseline-uninformed", "static-ml", "bandit", "bandit-adaptive"]

COLORS = {
    "baseline-uninformed": "#e74c3c",
    "static-ml":           "#2ecc71",
    "bandit":              "#3498db",
    "bandit-adaptive":     "#9b59b6",
}

SHOCK_QUERY = 50

def load_aggregated_results(base_dir: Path, num_runs: int) -> dict:
    aggregated_results = {config: [] for config in EXPERIMENT_ORDER}
    
    for i in range(1, num_runs + 1):
        run_dir = base_dir / f"run_{i}"
        if not run_dir.exists():
            print(f"Warning: {run_dir} not found. Skipping.")
            continue
            
        for config in EXPERIMENT_ORDER:
            res_file = run_dir / f"{config}_results.json"
            if res_file.exists():
                with open(res_file) as f:
                    run_data = json.load(f)
                    for r in run_data:
                        r["run_id"] = i
                    aggregated_results[config].extend(run_data)
                    
    # Remove empty configs
    return {k: v for k, v in aggregated_results.items() if v}

def get_latencies(data: list, phase: str = "full") -> list[float]:
    successful = [r for r in data if r.get("success")]
    if phase == "pre":
        successful = [r for r in successful if r.get("query_number", 0) < SHOCK_QUERY]
    elif phase == "post":
        successful = [r for r in successful if r.get("query_number", 0) >= SHOCK_QUERY]
    return [r["total_time_ms"] for r in successful]

def compute_aggregated_summaries(results: dict) -> dict:
    summaries = {}
    for config, data in results.items():
        successful = [r for r in data if r.get("success")]
        total_queries = len(data)
        success_count = len(successful)
        
        if success_count > 0:
            latencies = [r["total_time_ms"] for r in successful]
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            mean_lat = np.mean(latencies)
            std_lat = np.std(latencies, ddof=1) if n > 1 else 0
            
            summaries[config] = {
                "config": config,
                "total_queries": total_queries,
                "successful_queries": success_count,
                "success_rate": success_count / total_queries if total_queries > 0 else 0,
                "mean_latency_ms": float(mean_lat),
                "p50_latency_ms": sorted_latencies[int(n * 0.50)],
                "p95_latency_ms": sorted_latencies[min(int(n * 0.95), n-1)],
                "p99_latency_ms": sorted_latencies[min(int(n * 0.99), n-1)],
                "std_latency_ms": float(std_lat)
            }
    return summaries

def create_latency_comparison_chart(summaries: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in summaries]
    if not configs: return
    means = [summaries[c].get("mean_latency_ms", 0) for c in configs]
    stds = [summaries[c].get("std_latency_ms", 0) for c in configs]
    colors = [COLORS.get(c, "gray") for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Scheduler', fontsize=12)
    ax.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax.set_title('Aggregated Scheduler Performance - Mean End-to-End Latency', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in configs], fontsize=10)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.0f}+/-{std:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'latency_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: latency_comparison.png")

def create_percentile_comparison(summaries: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in summaries and summaries[c].get("p50_latency_ms")]
    if not configs: return
    p50s = [summaries[c]["p50_latency_ms"] for c in configs]
    p95s = [summaries[c]["p95_latency_ms"] for c in configs]
    p99s = [summaries[c]["p99_latency_ms"] for c in configs]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, p50s, width, label='P50', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x,         p95s, width, label='P95', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, p99s, width, label='P99', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Scheduler', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Aggregated Latency Percentiles by Scheduler (P50, P95, P99)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in configs], fontsize=10)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'percentile_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: percentile_comparison.png")

def create_latency_over_time_avg(results: dict, output_path: Path):
    configs = [c for c in EXPERIMENT_ORDER if c in results]
    if not configs: return

    fig, ax = plt.subplots(figsize=(12, 6))

    for config in configs:
        data = results[config]
        successful = [r for r in data if r.get("success")]
        if not successful: continue

        query_stats = {}
        for r in successful:
            q = r["query_number"]
            if q not in query_stats:
                query_stats[q] = []
            query_stats[q].append(r["total_time_ms"])
            
        queries = sorted(list(query_stats.keys()))
        mean_latencies = [np.mean(query_stats[q]) for q in queries]

        ax.plot(queries, mean_latencies, label=config, color=COLORS.get(config, 'gray'),
                alpha=0.8, linewidth=1.5)

    ax.axvline(x=SHOCK_QUERY, color='red', linestyle='--', alpha=0.6, label=f'Shock (Q{SHOCK_QUERY})')

    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Average Latency (ms) across all runs', fontsize=12)
    ax.set_title('Average Latency Over Time - Adaptive Behaviour Under Shock', fontsize=14)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'latency_over_time_avg.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: latency_over_time_avg.png")

def create_post_shock_convergence_avg(results: dict, output_path: Path):
    bandit_configs = [c for c in ["bandit", "bandit-adaptive"] if c in results]
    if not bandit_configs: return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for config in bandit_configs:
        data = results[config]
        successful = [r for r in data if r.get("success") and r.get("query_number", 0) >= SHOCK_QUERY]
        if not successful: continue

        query_stats = {}
        for r in successful:
            q = r["query_number"]
            if q not in query_stats:
                query_stats[q] = []
            query_stats[q].append(r["total_time_ms"])

        queries = sorted(list(query_stats.keys()))
        mean_latencies = [np.mean(query_stats[q]) for q in queries]
        
        cumulative_means = [np.mean(mean_latencies[:i+1]) for i in range(len(mean_latencies))]

        ax.plot(queries, cumulative_means, label=config,
                color=COLORS.get(config, 'gray'), linewidth=2)

    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Cumulative Mean Latency (ms) across runs', fontsize=12)
    ax.set_title(f'Aggregated Post-Shock Convergence (Q{SHOCK_QUERY}+)\nLower = faster recovery', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'post_shock_convergence_avg.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"saved: post_shock_convergence_avg.png")

def cohens_d(a: list, b: list) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return 0.0
    s1 = np.std(a, ddof=1)
    s2 = np.std(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0: return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)

def run_comparison(label: str, lat1: list, lat2: list, config1: str, config2: str,
                   n_comparisons: int) -> dict:
    if not lat1 or not lat2: return None

    t_stat, p_raw = stats.ttest_ind(lat1, lat2, equal_var=False)
    p_bonferroni = min(p_raw * n_comparisons, 1.0)
    d = cohens_d(lat1, lat2)

    if abs(d) < 0.2: effect = "negligible"
    elif abs(d) < 0.5: effect = "small"
    elif abs(d) < 0.8: effect = "medium"
    elif abs(d) < 1.2: effect = "large"
    else: effect = "very large"

    return {
        "phase": label,
        "config_1": config1,
        "config_2": config2,
        "n_1": len(lat1),
        "n_2": len(lat2),
        "mean_1": float(np.mean(lat1)),
        "mean_2": float(np.mean(lat2)),
        "std_1": float(np.std(lat1, ddof=1)),
        "std_2": float(np.std(lat2, ddof=1)),
        "t_statistic": float(t_stat),
        "p_raw": float(p_raw),
        "p_bonferroni": float(p_bonferroni),
        "significant_raw": bool(p_raw < 0.05),
        "significant_bonferroni": bool(p_bonferroni < 0.05),
        "cohens_d": float(d),
        "effect_size": effect,
        "improvement_pct": float((np.mean(lat1) - np.mean(lat2)) / np.mean(lat1) * 100) if np.mean(lat1) > 0 else 0.0
    }

def perform_statistical_tests(results: dict, output_path: Path) -> list:
    pairs = [
        ("baseline-uninformed", "static-ml"),
        ("baseline-uninformed", "bandit"),
        ("baseline-uninformed", "bandit-adaptive"),
        ("static-ml",           "bandit"),
        ("static-ml",           "bandit-adaptive"),
        ("bandit",              "bandit-adaptive"),
    ]
    n_comparisons = len(pairs)
    comparisons = []

    for config1, config2 in pairs:
        if config1 not in results or config2 not in results:
            continue

        for phase in ("full", "pre", "post"):
            lat1 = get_latencies(results[config1], phase)
            lat2 = get_latencies(results[config2], phase)
            phase_label = {"full": "full (q1-100)", "pre": "pre-shock (q1-49)", "post": "post-shock (q50-100)"}[phase]

            result = run_comparison(phase_label, lat1, lat2, config1, config2, n_comparisons)
            if result:
                comparisons.append(result)

    stats_file = output_path / 'aggregated_statistical_tests.json'
    with open(stats_file, 'w') as f:
        json.dump(comparisons, f, indent=2)
    print(f"saved: aggregated_statistical_tests.json")

    print("\n--- aggregated statistical analysis ---")
    print(f"bonferroni correction: alpha/n = 0.05/{n_comparisons} = {0.05/n_comparisons:.4f}\n")

    for c in comparisons:
        sig = ""
        if c["p_bonferroni"] < 0.001: sig = "*** (bonf)"
        elif c["p_bonferroni"] < 0.01: sig = "** (bonf)"
        elif c["p_bonferroni"] < 0.05: sig = "* (bonf)"
        elif c["p_raw"] < 0.05: sig = "* (uncorr only)"

        print(f"[{c['phase']}] {c['config_1']} vs {c['config_2']}")
        print(f"  {c['mean_1']:.1f}ms (sd={c['std_1']:.1f}, n={c['n_1']}) vs {c['mean_2']:.1f}ms (sd={c['std_2']:.1f}, n={c['n_2']})")
        print(f"  t={c['t_statistic']:.3f}, p={c['p_raw']:.4f}, p_bonf={c['p_bonferroni']:.4f} {sig}")
        print(f"  Cohen's d={c['cohens_d']:.3f} ({c['effect_size']}), improvement={c['improvement_pct']:+.1f}%")
        print()

    return comparisons

def create_summary_table(summaries: dict, results: dict, output_path: Path) -> pd.DataFrame:
    rows = []
    baseline_mean = summaries.get("baseline-uninformed", {}).get("mean_latency_ms", 0)

    for config in EXPERIMENT_ORDER:
        if config not in summaries: continue
        s = summaries[config]

        pre_mean, post_mean = None, None
        if config in results:
            pre = get_latencies(results[config], "pre")
            post = get_latencies(results[config], "post")
            if pre: pre_mean = round(np.mean(pre), 1)
            if post: post_mean = round(np.mean(post), 1)

        improvement = 0.0
        if baseline_mean > 0 and s.get("mean_latency_ms"):
            improvement = (baseline_mean - s["mean_latency_ms"]) / baseline_mean * 100

        rows.append({
            "Scheduler": config,
            "Total Samples": f"{s.get('successful_queries', 0)} / {s.get('total_queries', 0)}",
            "Mean (ms)": f"{s.get('mean_latency_ms', 0):.1f}",
            "Std (ms)": f"{s.get('std_latency_ms', 0):.1f}",
            "P95 (ms)": f"{s.get('p95_latency_ms', 0):.1f}",
            "P99 (ms)": f"{s.get('p99_latency_ms', 0):.1f}",
            "Pre-shock mean": f"{pre_mean}" if pre_mean is not None else "n/a",
            "Post-shock mean": f"{post_mean}" if post_mean is not None else "n/a",
            "vs baseline": f"{improvement:+.1f}%"
        })

    df = pd.DataFrame(rows)
    csv_path = output_path / 'aggregated_comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"saved: aggregated_comparison_table.csv")

    print("\n--- aggregated comparison table ---")
    print(df.to_string(index=False))
    return df

def main():
    parser = argparse.ArgumentParser(description="analyze aggregated scheduler experiment results across multiple runs")
    parser.add_argument("--input", type=str, default="results", help="base results directory containing run_1, run_2, etc.")
    parser.add_argument("--runs", type=int, default=10, help="number of runs to aggregate")
    parser.add_argument("--output", type=str, default=None, help="output directory for aggregated analysis")
    args = parser.parse_args()

    input_dir = Path(args.input)
    num_runs = args.runs
    output_dir = Path(args.output) if args.output else input_dir / "aggregated_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading aggregated results from {num_runs} runs in: {input_dir}")
    results = load_aggregated_results(input_dir, num_runs)

    if not results:
        print("no results found")
        sys.exit(1)

    print(f"found data for: {list(results.keys())}")
    
    summaries = compute_aggregated_summaries(results)

    create_latency_comparison_chart(summaries, output_dir)
    create_percentile_comparison(summaries, output_dir)
    create_latency_over_time_avg(results, output_dir)
    create_post_shock_convergence_avg(results, output_dir)

    perform_statistical_tests(results, output_dir)
    create_summary_table(summaries, results, output_dir)

    print(f"\ndone - aggregated results in {output_dir}")

if __name__ == "__main__":
    main()
