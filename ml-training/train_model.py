#!/usr/bin/env python3
# Trains the Random Forest for `static` routing mode, writes gateway_model.pkl + diagnostics.
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
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent
FEATURE_NAMES = ["prompt_length", "gpu_in_flight", "cpu_in_flight"]


def load_dataset(path: Path):
    df = pd.read_csv(path)
    X = df[FEATURE_NAMES]
    y = df["optimal_node"]
    return X, y


def train(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def plot_feature_importance(model, output_path: Path):
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
    bars = ax.barh(range(len(features)), scores, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance Score")
    ax.set_title("Gateway Static Router — Feature Importance\n(Random Forest)")
    ax.invert_yaxis()
    for bar, score in zip(bars, scores):
        ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {output_path.name}")


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["GPU", "CPU"],
                yticklabels=["GPU", "CPU"],
                annot_kws={"size": 14}, ax=ax)
    ax.set_title("Gateway Static Router — Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {output_path.name}")


def main():
    print("Gateway static router model training")

    data_path = OUTPUT_DIR / "training_data.csv"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_dataset.py first.")
        return

    X, y = load_dataset(data_path)
    print(f"Loaded {len(X)} samples | features: {FEATURE_NAMES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nTraining Random Forest...")
    model = train(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\nTest accuracy : {acc:.4f}")
    print(f"F1 (weighted) : {f1:.4f}")
    print(f"ROC AUC       : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['GPU', 'CPU'])}")

    print("Feature importance:")
    for feat, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"  {feat:20} {imp:.4f}  {bar}")

    model_path = OUTPUT_DIR / "gateway_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nsaved: {model_path.name}")

    report = {
        "model_type": "RandomForestClassifier",
        "feature_names": FEATURE_NAMES,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "feature_importance": {f: float(i) for f, i in zip(FEATURE_NAMES, model.feature_importances_)},
        "confusion_matrix": cm.tolist(),
    }
    report_path = OUTPUT_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {report_path.name}")

    plot_feature_importance(model, OUTPUT_DIR / "feature_importance.png")
    plot_confusion_matrix(cm, OUTPUT_DIR / "confusion_matrix.png")

    print("\nTraining complete")
    print(f"Model: {model_path}")
    print("\nNext: copy gateway_model.pkl into the smart-gateway Docker image")
    print("      (or mount it via a volume and set STATIC_MODEL_PATH env var)")


if __name__ == "__main__":
    main()
