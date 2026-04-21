#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent


def generate_dataset(n_samples: int = 3000, noise_rate: float = 0.05) -> pd.DataFrame:
    records = []

    for _ in range(n_samples):
        if np.random.random() < 0.6:
            prompt_length = int(np.random.uniform(20, 80))
        else:
            prompt_length = int(np.random.uniform(80, 300))

        gpu_in_flight = int(np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
        ))
        cpu_in_flight = int(np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            p=[0.35, 0.25, 0.15, 0.09, 0.06, 0.04, 0.03, 0.01, 0.01, 0.005, 0.005]
        ))

        if gpu_in_flight > 8:
            optimal_node = 1
        elif cpu_in_flight > 8:
            optimal_node = 0
        elif gpu_in_flight > 5 and prompt_length > 100:
            optimal_node = 1
        elif gpu_in_flight > 5 and prompt_length <= 100:
            optimal_node = 0
        else:
            optimal_node = 0

        if np.random.random() < noise_rate:
            optimal_node = 1 - optimal_node

        records.append({
            "prompt_length":  prompt_length,
            "gpu_in_flight":  gpu_in_flight,
            "cpu_in_flight":  cpu_in_flight,
            "optimal_node":   optimal_node,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate gateway routing training data")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--noise",   type=float, default=0.05)
    parser.add_argument("--output",  type=str, default="training_data.csv")
    args = parser.parse_args()

    print("Gateway routing dataset generator")
    print(f"Features: prompt_length, gpu_in_flight, cpu_in_flight")
    print(f"Label:    optimal_node (0=GPU, 1=CPU)")

    df = generate_dataset(n_samples=args.samples, noise_rate=args.noise)

    output_path = OUTPUT_DIR / args.output
    df.to_csv(output_path, index=False)

    n = len(df)
    gpu_count = (df["optimal_node"] == 0).sum()
    cpu_count = (df["optimal_node"] == 1).sum()

    print(f"\nGenerated {n} samples -> {output_path}")
    print(f"  GPU (0): {gpu_count} ({gpu_count/n*100:.1f}%)")
    print(f"  CPU (1): {cpu_count} ({cpu_count/n*100:.1f}%)")

    df["gpu_load_band"] = pd.cut(
        df["gpu_in_flight"],
        bins=[-1, 2, 5, 8, 99],
        labels=["low (0-2)", "medium (3-5)", "high (6-8)", "critical (>8)"]
    )
    xtab = pd.crosstab(df["gpu_load_band"], df["optimal_node"].map({0: "GPU", 1: "CPU"}))
    print(f"\nRouting by GPU queue depth:\n{xtab.to_string()}")

    stats = {
        "total_samples": n,
        "gpu_samples": int(gpu_count),
        "cpu_samples": int(cpu_count),
        "gpu_pct": round(gpu_count / n * 100, 1),
        "cpu_pct": round(cpu_count / n * 100, 1),
    }
    with open(OUTPUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNext: python train_model.py")


if __name__ == "__main__":
    main()
