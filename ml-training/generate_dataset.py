#!/usr/bin/env python3
"""
ml-training/generate_dataset.py

Generate synthetic training data for the L7 Smart Gateway static router.

Unlike the old pod-level static scheduler (which used task_type, model_size, etc.),
the gateway can only observe what is visible at request time:
    - prompt_length   : word count of the full augmented prompt
    - gpu_in_flight   : current requests in-flight to GPU Ollama
    - cpu_in_flight   : current requests in-flight to CPU Ollama

Label: 0 = route to GPU, 1 = route to CPU

Expert routing rules encoded in the training data:
    1. GPU is the fast path — always prefer it when not overloaded.
    2. If gpu_in_flight > 5 AND prompt is long (>100 words): route to CPU.
       Long prompts take 30-60s on GPU when there are 5+ competing requests;
       CPU finishes them faster even without acceleration.
    3. If gpu_in_flight > 8: always route to CPU (GPU critically saturated).
    4. If cpu_in_flight > 8: route to GPU even if it is somewhat loaded.
    5. Add 5% label noise to simulate imperfect expert knowledge.
"""

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
        # Prompt length: most queries are short, some are long (RAG context inflates them)
        # Short: 20-80 words (bare question), Long: 80-300 words (with context docs injected)
        if np.random.random() < 0.6:
            prompt_length = int(np.random.uniform(20, 80))
        else:
            prompt_length = int(np.random.uniform(80, 300))

        # Queue depths: simulate bursty traffic patterns
        # GPU queue: often 0-4 but spikes to 10+ under load
        gpu_in_flight = int(np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
            p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.07, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
        ))
        cpu_in_flight = int(np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            p=[0.35, 0.25, 0.15, 0.09, 0.06, 0.04, 0.03, 0.01, 0.01, 0.005, 0.005]
        ))

        # ── Expert scheduling logic ────────────────────────────────────────────
        if gpu_in_flight > 8:
            # GPU critically saturated — always offload regardless of prompt size
            optimal_node = 1  # CPU

        elif cpu_in_flight > 8:
            # CPU also saturated — GPU is the lesser evil
            optimal_node = 0  # GPU

        elif gpu_in_flight > 5 and prompt_length > 100:
            # GPU moderately overloaded + long prompt: CPU wins
            # (long prompts compete badly with 5+ in-flight GPU requests)
            optimal_node = 1  # CPU

        elif gpu_in_flight > 5 and prompt_length <= 100:
            # GPU moderately overloaded but prompt is short:
            # Short prompts finish quickly even on a loaded GPU, stay there
            optimal_node = 0  # GPU

        else:
            # Normal case: GPU is preferred (faster for any prompt length)
            optimal_node = 0  # GPU

        # ── Label noise ────────────────────────────────────────────────────────
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

    print("=" * 60)
    print("GATEWAY ROUTING DATASET GENERATOR")
    print(f"Features: prompt_length, gpu_in_flight, cpu_in_flight")
    print(f"Label:    optimal_node (0=GPU, 1=CPU)")
    print("=" * 60)

    df = generate_dataset(n_samples=args.samples, noise_rate=args.noise)

    output_path = OUTPUT_DIR / args.output
    df.to_csv(output_path, index=False)

    n = len(df)
    gpu_count = (df["optimal_node"] == 0).sum()
    cpu_count = (df["optimal_node"] == 1).sum()

    print(f"\nGenerated {n} samples -> {output_path}")
    print(f"  GPU (0): {gpu_count} ({gpu_count/n*100:.1f}%)")
    print(f"  CPU (1): {cpu_count} ({cpu_count/n*100:.1f}%)")

    # Cross-tab: queue depth vs routing decision
    df["gpu_load_band"] = pd.cut(
        df["gpu_in_flight"],
        bins=[-1, 2, 5, 8, 99],
        labels=["low (0-2)", "medium (3-5)", "high (6-8)", "critical (>8)"]
    )
    xtab = pd.crosstab(df["gpu_load_band"], df["optimal_node"].map({0: "GPU", 1: "CPU"}))
    print(f"\nRouting by GPU queue depth:\n{xtab.to_string()}")

    # Save stats
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
