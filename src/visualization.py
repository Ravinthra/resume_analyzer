"""
visualization.py — Evaluation Visualization Module
=====================================================

Generates plots for evaluation reports:
  - Confusion matrix heatmap (25×25)
  - Class distribution bar chart
  - Per-class F1 bar chart
  - Similarity distribution histogram

Author: AI Resume Analyzer Project
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional


def setup_style():
    """Configure plot style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
):
    """Plot confusion matrix heatmap."""
    setup_style()
    fig, ax = plt.subplots(figsize=(16, 14))

    # Normalize
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, cbar_kws={"label": "Proportion"}
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_class_distribution(
    labels: List[int],
    label_names: List[str],
    save_path: str,
    title: str = "Class Distribution"
):
    """Plot class distribution bar chart."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    counts = {}
    for label in labels:
        name = label_names[label] if label < len(label_names) else str(label)
        counts[name] = counts.get(name, 0) + 1

    names = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")

    # Add count labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_per_class_f1(
    report: Dict,
    save_path: str,
    title: str = "Per-Class F1 Score"
):
    """Plot F1 score for each class."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract per-class F1 scores
    classes = []
    f1_scores = []
    for key, val in report.items():
        if key not in ("accuracy", "macro avg", "weighted avg") and isinstance(val, dict):
            classes.append(key)
            f1_scores.append(val.get("f1-score", 0))

    # Sort by F1
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1], reverse=True)
    classes = [p[0] for p in sorted_pairs]
    f1_scores = [p[1] for p in sorted_pairs]

    colors = ["#2ecc71" if f >= 0.9 else "#f39c12" if f >= 0.7 else "#e74c3c" for f in f1_scores]

    bars = ax.barh(classes, f1_scores, color=colors, edgecolor="white")
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5, label="Target (0.90)")
    ax.legend()

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                f"{val:.3f}", ha="left", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_similarity_distribution(
    correct_scores: List[float],
    incorrect_scores: List[float],
    save_path: str,
    title: str = "Similarity Score Distribution"
):
    """Plot histogram of correct vs incorrect match similarity scores."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    if correct_scores:
        ax.hist(correct_scores, bins=30, alpha=0.6, color="#2ecc71",
                label=f"Correct matches (n={len(correct_scores)})", density=True)
    if incorrect_scores:
        ax.hist(incorrect_scores, bins=30, alpha=0.6, color="#e74c3c",
                label=f"Incorrect matches (n={len(incorrect_scores)})", density=True)

    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {save_path}")
