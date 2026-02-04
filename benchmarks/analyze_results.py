#!/usr/bin/env python3
"""
benchmarks/analyze_results.py

Analyze experiment results and generate publication-quality visualizations.

Usage:
    python analyze_results.py --input results/ --output results/analysis/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)


def load_results(results_dir: Path) -> dict:
    """Load all experiment results from directory."""
    results = {}
    summaries = {}
    
    for file in results_dir.glob("*_results.json"):
        config = file.stem.replace("_results", "")
        with open(file) as f:
            results[config] = json.load(f)
    
    for file in results_dir.glob("*_summary.json"):
        config = file.stem.replace("_summary", "")
        with open(file) as f:
            summaries[config] = json.load(f)
    
    return results, summaries


def create_latency_comparison_chart(summaries: dict, output_path: Path):
    """Create bar chart comparing mean latency across schedulers."""
    configs = []
    means = []
    stds = []
    
    # Order configurations logically
    order = [
        "baseline-uninformed",
        "baseline-informed", 
        "baseline-optimal",
        "static-ml",
        "bandit-adaptive"
    ]
    
    for config in order:
        if config in summaries:
            s = summaries[config]
            configs.append(config)
            means.append(s.get("mean_latency_ms", 0))
            stds.append(s.get("std_latency_ms", 0))
    
    if not configs:
        print("No data for latency comparison chart")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    x = np.arange(len(configs))
    colors = sns.color_palette("husl", len(configs))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Scheduler Configuration', fontsize=12)
    ax.set_ylabel('Mean Latency (ms)', fontsize=12)
    ax.set_title('Scheduler Performance Comparison\nMean End-to-End Latency', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in configs], fontsize=10)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.0f}±{std:.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'latency_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path / 'latency_comparison.png'}")


def create_percentile_comparison(summaries: dict, output_path: Path):
    """Create grouped bar chart for P50, P95, P99 latencies."""
    configs = []
    p50s = []
    p95s = []
    p99s = []
    
    order = ["baseline-uninformed", "baseline-informed", "baseline-optimal", "static-ml", "bandit-adaptive"]
    
    for config in order:
        if config in summaries:
            s = summaries[config]
            if s.get("p50_latency_ms"):
                configs.append(config)
                p50s.append(s["p50_latency_ms"])
                p95s.append(s["p95_latency_ms"])
                p99s.append(s["p99_latency_ms"])
    
    if not configs:
        print("No data for percentile comparison")
        return
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    x = np.arange(len(configs))
    width = 0.25
    
    bars1 = ax.bar(x - width, p50s, width, label='P50', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, p95s, width, label='P95', color='#f39c12', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, p99s, width, label='P99', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Scheduler Configuration', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Percentiles by Scheduler\n(P50, P95, P99)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in configs], fontsize=10)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'percentile_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path / 'percentile_comparison.png'}")


def create_latency_over_time(results: dict, output_path: Path, configs: list = None):
    """Create line plot showing latency over query number."""
    if configs is None:
        configs = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'static-ml': '#2ecc71', 'bandit-adaptive': '#9b59b6', 
              'baseline-optimal': '#3498db', 'baseline-informed': '#f39c12',
              'baseline-uninformed': '#e74c3c'}
    
    for config in configs:
        if config not in results:
            continue
        
        data = results[config]
        successful = [r for r in data if r.get("success")]
        
        if not successful:
            continue
        
        queries = [r["query_number"] for r in successful]
        latencies = [r["total_time_ms"] for r in successful]
        
        color = colors.get(config, 'gray')
        ax.plot(queries, latencies, label=config, color=color, alpha=0.8, linewidth=1.5)
    
    # Mark shock point if visible
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Shock Event (Q50)')
    
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Over Time\n(Demonstrates Adaptive Behavior Under Shock)', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path / 'latency_over_time.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path / 'latency_over_time.png'}")


def create_cumulative_regret(results: dict, output_path: Path, optimal_config: str = "baseline-optimal"):
    """Create cumulative regret plot comparing schedulers."""
    if optimal_config not in results:
        print(f"Optimal config '{optimal_config}' not found, skipping regret plot")
        return
    
    # Get optimal latencies
    optimal_data = results[optimal_config]
    optimal_successful = [r for r in optimal_data if r.get("success")]
    optimal_latencies = {r["query_number"]: r["total_time_ms"] for r in optimal_successful}
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    colors = {'static-ml': '#2ecc71', 'bandit-adaptive': '#9b59b6'}
    
    for config in ['static-ml', 'bandit-adaptive']:
        if config not in results:
            continue
        
        data = results[config]
        successful = [r for r in data if r.get("success")]
        
        # Calculate regret
        cumulative_regret = []
        total_regret = 0
        
        for r in sorted(successful, key=lambda x: x["query_number"]):
            q = r["query_number"]
            actual_latency = r["total_time_ms"]
            optimal_latency = optimal_latencies.get(q, actual_latency)
            
            regret = max(0, actual_latency - optimal_latency)
            total_regret += regret
            cumulative_regret.append((q, total_regret))
        
        if cumulative_regret:
            queries, regrets = zip(*cumulative_regret)
            ax.plot(queries, regrets, label=config, color=colors.get(config, 'gray'), linewidth=2)
    
    # Mark shock point
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='Shock Event')
    
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Cumulative Regret (ms)', fontsize=12)
    ax.set_title('Cumulative Regret Over Time\n(Lower is Better)', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'cumulative_regret.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path / 'cumulative_regret.png'}")


def perform_statistical_tests(summaries: dict, results: dict, output_path: Path):
    """Perform t-tests comparing schedulers and save results."""
    comparisons = []
    
    # Compare each pair
    pairs = [
        ("baseline-uninformed", "static-ml"),
        ("baseline-uninformed", "bandit-adaptive"),
        ("baseline-informed", "static-ml"),
        ("static-ml", "bandit-adaptive"),
        ("baseline-optimal", "static-ml"),
    ]
    
    for config1, config2 in pairs:
        if config1 not in results or config2 not in results:
            continue
        
        # Get latencies
        lat1 = [r["total_time_ms"] for r in results[config1] if r.get("success")]
        lat2 = [r["total_time_ms"] for r in results[config2] if r.get("success")]
        
        if not lat1 or not lat2:
            continue
        
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(lat1, lat2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(lat1)**2 + np.std(lat2)**2) / 2)
        cohens_d = (np.mean(lat1) - np.mean(lat2)) / pooled_std if pooled_std > 0 else 0
        
        comparison = {
            "config_1": config1,
            "config_2": config2,
            "mean_1": np.mean(lat1),
            "mean_2": np.mean(lat2),
            "std_1": np.std(lat1),
            "std_2": np.std(lat2),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "improvement_pct": (np.mean(lat1) - np.mean(lat2)) / np.mean(lat1) * 100
        }
        comparisons.append(comparison)
    
    # Save results
    stats_file = output_path / 'statistical_tests.json'
    with open(stats_file, 'w') as f:
        json.dump(comparisons, f, indent=2)
    print(f"✓ Saved: {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    for c in comparisons:
        sig = "***" if c["p_value"] < 0.001 else "**" if c["p_value"] < 0.01 else "*" if c["p_value"] < 0.05 else ""
        print(f"\n{c['config_1']} vs {c['config_2']}:")
        print(f"  Mean diff: {c['mean_1']:.1f}ms vs {c['mean_2']:.1f}ms ({c['improvement_pct']:+.1f}%)")
        print(f"  t={c['t_statistic']:.3f}, p={c['p_value']:.4f} {sig}")
        print(f"  Cohen's d: {c['cohens_d']:.3f}")
    
    return comparisons


def create_summary_table(summaries: dict, output_path: Path):
    """Create summary comparison table."""
    rows = []
    
    order = ["baseline-uninformed", "baseline-informed", "baseline-optimal", "static-ml", "bandit-adaptive"]
    
    for config in order:
        if config not in summaries:
            continue
        s = summaries[config]
        
        # Calculate improvement vs uninformed baseline
        baseline = summaries.get("baseline-uninformed", {}).get("mean_latency_ms", s.get("mean_latency_ms", 0))
        improvement = (baseline - s.get("mean_latency_ms", 0)) / baseline * 100 if baseline > 0 else 0
        
        rows.append({
            "Configuration": config,
            "Mean Latency (ms)": f"{s.get('mean_latency_ms', 0):.1f}",
            "Std Dev (ms)": f"{s.get('std_latency_ms', 0):.1f}",
            "P95 (ms)": f"{s.get('p95_latency_ms', 0):.1f}",
            "P99 (ms)": f"{s.get('p99_latency_ms', 0):.1f}",
            "Success Rate": f"{s.get('success_rate', 0)*100:.1f}%",
            "Improvement vs Baseline": f"{improvement:+.1f}%"
        })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = output_path / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Print
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--input", type=str, default="results", help="Input results directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for analysis")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    results, summaries = load_results(input_dir)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    print(f"Found {len(results)} experiment configurations: {list(results.keys())}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_latency_comparison_chart(summaries, output_dir)
    create_percentile_comparison(summaries, output_dir)
    create_latency_over_time(results, output_dir)
    create_cumulative_regret(results, output_dir)
    
    # Statistical analysis
    perform_statistical_tests(summaries, results, output_dir)
    
    # Summary table
    create_summary_table(summaries, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
