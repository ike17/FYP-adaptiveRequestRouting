#!/usr/bin/env python3
"""
ml-schedulers/static/generate_dataset.py

Generate synthetic training dataset for the Static ML Scheduler.
This simulates expert scheduling decisions for a heterogeneous cluster.

The dataset represents the "ground truth" that an expert human scheduler
would decide, allowing us to train a model to replicate these decisions.

Features:
    - task_type: 0=retrieval (CPU-optimal), 1=generation (GPU-optimal)
    - input_size_mb: Size of input data (affects processing time)
    - model_size_b: LLM parameter count in billions (0 for retrieval tasks)
    - cpu_node_load: Current CPU node utilization [0-1]
    - gpu_node_load: Current GPU node utilization [0-1]
    - memory_required_gb: Pod memory request
    - is_batch_request: 0=single query, 1=batch processing

Label:
    - optimal_node: 0=CPU node (minikube), 1=GPU node (minikube-m02)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json

# Reproducibility
np.random.seed(42)


def generate_scheduling_dataset(n_samples: int = 2000, noise_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic scheduling dataset with expert labels.
    
    Args:
        n_samples: Number of scheduling scenarios to generate
        noise_rate: Fraction of labels to flip (simulates imperfect expert knowledge)
    
    Returns:
        DataFrame with features and optimal_node labels
    """
    data = []
    
    for i in range(n_samples):
        # =================================================================
        # FEATURE GENERATION
        # =================================================================
        
        # Task type: 40% retrieval, 60% generation (reflects RAG workload mix)
        task_type = np.random.choice([0, 1], p=[0.4, 0.6])
        
        if task_type == 0:  # Retrieval task (CPU-optimal)
            input_size_mb = np.random.uniform(1, 100)
            model_size_b = 0.0  # No LLM for retrieval
            memory_required_gb = np.random.uniform(0.5, 2.0)
            is_batch = np.random.choice([0, 1], p=[0.8, 0.2])
        else:  # Generation task (GPU-optimal)
            input_size_mb = np.random.uniform(50, 2000)
            # Model sizes: 2B (70%), 7B (20%), 13B (10%)
            model_size_b = np.random.choice([2.0, 7.0, 13.0], p=[0.7, 0.2, 0.1])
            memory_required_gb = np.random.uniform(2.0, 8.0)
            is_batch = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Cluster state: node utilization
        # Beta distributions create realistic load patterns
        cpu_node_load = np.random.beta(2, 5)  # Skewed low (mean ~0.29)
        gpu_node_load = np.random.beta(2, 3)  # Higher average (mean ~0.40)
        
        # =================================================================
        # EXPERT SCHEDULING LOGIC (Ground Truth)
        # =================================================================
        
        if task_type == 1:  # Generation task
            # GPU is strongly preferred for generation
            if gpu_node_load > 0.9 and cpu_node_load < 0.5:
                # GPU overloaded, CPU available
                # Only fallback for small models
                optimal_node = 0 if model_size_b <= 2.0 else 1
            elif gpu_node_load > 0.95:
                # GPU critically overloaded, must fallback
                optimal_node = 0
            else:
                # Normal case: GPU is optimal
                optimal_node = 1
        else:  # Retrieval task
            # CPU is preferred for retrieval
            if cpu_node_load > 0.85 and gpu_node_load < 0.3:
                # CPU overloaded, GPU idle - can overflow retrieval to GPU
                optimal_node = 1
            else:
                # Normal case: CPU is optimal
                optimal_node = 0
        
        # =================================================================
        # LABEL NOISE (Simulates imperfect expert knowledge)
        # =================================================================
        if np.random.random() < noise_rate:
            optimal_node = 1 - optimal_node
        
        # Store sample
        data.append({
            'task_type': task_type,
            'input_size_mb': round(input_size_mb, 2),
            'model_size_b': model_size_b,
            'cpu_node_load': round(cpu_node_load, 4),
            'gpu_node_load': round(gpu_node_load, 4),
            'memory_required_gb': round(memory_required_gb, 2),
            'is_batch_request': is_batch,
            'optimal_node': optimal_node
        })
    
    return pd.DataFrame(data)


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Generate statistics about the dataset."""
    stats = {
        'total_samples': len(df),
        'class_distribution': df['optimal_node'].value_counts().to_dict(),
        'task_type_distribution': df['task_type'].value_counts().to_dict(),
        'feature_correlations': df.corr()['optimal_node'].drop('optimal_node').to_dict()
    }
    
    # Cross-tabulation: task_type vs optimal_node
    crosstab = pd.crosstab(
        df['task_type'].map({0: 'Retrieval', 1: 'Generation'}),
        df['optimal_node'].map({0: 'CPU Node', 1: 'GPU Node'})
    )
    stats['task_node_crosstab'] = crosstab.to_dict()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic scheduling dataset')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--noise', type=float, default=0.05, help='Label noise rate')
    parser.add_argument('--output', type=str, default='training_data.csv', help='Output filename')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC DATASET GENERATOR")
    print("Static ML Scheduler Training Data")
    print("=" * 60)
    
    # Generate dataset
    print(f"\nGenerating {args.samples} samples with {args.noise*100:.1f}% noise...")
    df = generate_scheduling_dataset(n_samples=args.samples, noise_rate=args.noise)
    
    # Save dataset
    output_path = Path(__file__).parent / args.output
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")
    
    # Analyze and display statistics
    stats = analyze_dataset(df)
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nTotal samples: {stats['total_samples']}")
    
    print(f"\nClass distribution (optimal_node):")
    for node, count in stats['class_distribution'].items():
        node_name = "CPU Node" if node == 0 else "GPU Node"
        pct = count / stats['total_samples'] * 100
        print(f"  {node_name}: {count} ({pct:.1f}%)")
    
    print(f"\nTask type distribution:")
    for task, count in stats['task_type_distribution'].items():
        task_name = "Retrieval" if task == 0 else "Generation"
        pct = count / stats['total_samples'] * 100
        print(f"  {task_name}: {count} ({pct:.1f}%)")
    
    print(f"\nFeature correlations with optimal_node:")
    for feature, corr in sorted(stats['feature_correlations'].items(), 
                                 key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {corr:+.4f}")
    
    print(f"\nCross-tabulation (Task Type × Optimal Node):")
    crosstab_df = pd.DataFrame(stats['task_node_crosstab'])
    print(crosstab_df.to_string())
    
    # Save statistics
    stats_path = Path(__file__).parent / 'dataset_stats.json'
    # Convert numpy types for JSON serialization
    stats_serializable = {
        k: {str(kk): int(vv) if isinstance(vv, (np.integer, int)) else float(vv) 
            for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in stats.items()
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"\n✓ Statistics saved to: {stats_path}")
    
    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print("Next step: Run 'python train_model.py' to train the classifier")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
