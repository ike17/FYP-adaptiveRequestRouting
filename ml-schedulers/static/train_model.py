#!/usr/bin/env python3
"""
ml-schedulers/static/train_model.py

Train a Random Forest classifier for the Static ML Scheduler.
The model predicts optimal node placement (CPU vs GPU) based on pod features.

Outputs:
    - static_scheduler_model.pkl: Trained model
    - feature_names.pkl: Feature names for inference
    - feature_importance.png: Visualization
    - confusion_matrix.png: Model evaluation
    - training_report.json: Metrics and hyperparameters
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path(__file__).parent


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split dataset into features and labels."""
    df = pd.read_csv(path)
    X = df.drop('optimal_node', axis=1)
    y = df['optimal_node']
    return X, y


def train_model(X_train, y_train, tune_hyperparameters: bool = False):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: If True, perform grid search (slower but better)
    
    Returns:
        Trained model and best hyperparameters
    """
    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning (this may take a few minutes)...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    else:
        # Use sensible defaults for faster training
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle any class imbalance
        )
        
        model.fit(X_train, y_train)
        
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt'
        }
        
        return model, params


def evaluate_model(model, X_test, y_test, feature_names: list) -> dict:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary with all metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    
    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    metrics['feature_importance'] = importance
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=['CPU Node', 'GPU Node'],
                                   output_dict=True)
    metrics['classification_report'] = report
    
    return metrics


def plot_feature_importance(importance: dict, output_path: Path):
    """Create and save feature importance bar chart."""
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_features)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
    
    bars = plt.barh(range(len(features)), scores, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title('Static ML Scheduler - Feature Importance\n(Random Forest)')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for bar, score in zip(bars, scores):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance plot saved: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    """Create and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['CPU Node', 'GPU Node'],
                yticklabels=['CPU Node', 'GPU Node'],
                annot_kws={'size': 14})
    
    plt.title('Static ML Scheduler - Confusion Matrix\n(Test Set)', fontsize=12)
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix plot saved: {output_path}")


def main():
    print("=" * 60)
    print("STATIC ML SCHEDULER - MODEL TRAINING")
    print("Random Forest Classifier for Node Placement")
    print("=" * 60)
    
    # Load dataset
    data_path = OUTPUT_DIR / 'training_data.csv'
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        print("Run 'python generate_dataset.py' first!")
        return
    
    print(f"\nLoading dataset from: {data_path}")
    X, y = load_dataset(data_path)
    feature_names = X.columns.tolist()
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {feature_names}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\n" + "-" * 40)
    print("TRAINING")
    print("-" * 40)
    
    model, params = train_model(X_train, y_train, tune_hyperparameters=False)
    print(f"\n✓ Model trained with parameters: {params}")
    
    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Evaluate on test set
    print("\n" + "-" * 40)
    print("EVALUATION (Test Set)")
    print("-" * 40)
    
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nClassification Report:")
    print("-" * 40)
    report = metrics['classification_report']
    for label in ['CPU Node', 'GPU Node']:
        r = report[label]
        print(f"{label:12} precision:{r['precision']:.3f}  recall:{r['recall']:.3f}  f1:{r['f1-score']:.3f}")
    
    print("\nFeature Importance:")
    print("-" * 40)
    for feature, importance in sorted(metrics['feature_importance'].items(),
                                       key=lambda x: x[1], reverse=True):
        bar = "█" * int(importance * 40)
        print(f"{feature:20} {importance:.4f} {bar}")
    
    # Save model artifacts
    print("\n" + "-" * 40)
    print("SAVING ARTIFACTS")
    print("-" * 40)
    
    model_path = OUTPUT_DIR / 'static_scheduler_model.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    feature_path = OUTPUT_DIR / 'feature_names.pkl'
    joblib.dump(feature_names, feature_path)
    print(f"✓ Feature names saved: {feature_path}")
    
    # Generate visualizations
    plot_feature_importance(
        metrics['feature_importance'],
        OUTPUT_DIR / 'feature_importance.png'
    )
    
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        OUTPUT_DIR / 'confusion_matrix.png'
    )
    
    # Save training report
    report_data = {
        'model_type': 'RandomForestClassifier',
        'hyperparameters': params,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'test_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc'])
        },
        'feature_importance': {k: float(v) for k, v in metrics['feature_importance'].items()},
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    report_path = OUTPUT_DIR / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"✓ Training report saved: {report_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel artifacts saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Build scheduler Docker image")
    print("  2. Deploy to Kubernetes cluster")
    print("  3. Run experiments")


if __name__ == "__main__":
    main()
