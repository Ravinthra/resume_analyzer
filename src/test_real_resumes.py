"""
test_real_resumes.py — Real-World Evaluation Runner
=====================================================

Runs the full evaluation pipeline:
  1. Loads trained model
  2. Loads real-world test set (15K resumes)
  3. Evaluates classifier (accuracy, F1, confusion matrix)
  4. Runs error analysis
  5. Generates visualizations
  6. Produces evaluation report (reports/evaluation_report.md)

Usage:
    python -m src.test_real_resumes
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from src.model import get_model
from src.dataset import create_train_val_test_loaders
from src.evaluation import evaluate_classifier, analyze_errors
from src.visualization import (
    plot_confusion_matrix, plot_class_distribution,
    plot_per_class_f1
)


def main():
    start = time.time()
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("REAL-WORLD EVALUATION PIPELINE")
    print("=" * 60)

    # --- Load labels ---
    labels_path = project_root / "data" / "metadata" / "labels.json"
    if not labels_path.exists():
        labels_path = project_root / "data" / "labels.json"
    with open(labels_path, "r") as f:
        labels_data = json.load(f)
    num_classes = labels_data["num_classes"]
    label_names = [labels_data["id_to_label"][str(i)] for i in range(num_classes)]
    print(f"[Labels] {num_classes} classes loaded")

    # --- Load data (3-way split) ---
    print("\n[1/5] Loading data...")
    csv_path = str(project_root / "data" / "metadata" / "dataset.csv")
    _, _, test_loader = create_train_val_test_loaders(
        csv_path=csv_path,
        project_root=str(project_root),
        batch_size=8,
        max_length=256
    )

    # --- Load trained model ---
    print("\n[2/5] Loading model...")
    model_path = project_root / "models" / "bert_classifier.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists():
        print(f"[ERROR] No trained model found at {model_path}")
        print("Please run training first: python -m src.train")
        return

    model, device = get_model(num_classes=num_classes, dropout_rate=0.3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded from {model_path} (device: {device})")

    # --- Evaluate classifier ---
    print("\n[3/5] Evaluating classifier on test set...")
    results = evaluate_classifier(model, test_loader, device, label_names)

    print(f"\n   Accuracy:       {results['accuracy']:.4f}")
    print(f"   Macro Precision: {results['precision']:.4f}")
    print(f"   Macro Recall:    {results['recall']:.4f}")
    print(f"   Macro F1:        {results['f1']:.4f}")
    print(f"   Top-3 Accuracy:  {results['top3_accuracy']:.4f}")

    # --- Error analysis ---
    print("\n[4/5] Running error analysis...")
    errors = analyze_errors(
        results["predictions"], results["labels"],
        label_names, results["logits"]
    )
    print(f"   Total errors: {errors['total_errors']} / {len(results['labels'])}")
    print(f"   Error rate: {errors['error_rate']:.4f}")

    if errors["top_confusion_pairs"]:
        print("\n   Top confusion pairs:")
        for pair, count in errors["top_confusion_pairs"][:5]:
            print(f"     {pair}: {count}")

    # --- Generate visualizations ---
    print("\n[5/5] Generating visualizations...")

    # Confusion matrix
    unique_labels = sorted(set(results["labels"]))
    cm_labels = [label_names[i] for i in unique_labels]
    plot_confusion_matrix(
        results["confusion_matrix"], cm_labels,
        str(reports_dir / "confusion_matrix.png"),
        "Confusion Matrix — Real-World Test Set"
    )

    # Class distribution
    plot_class_distribution(
        results["labels"], label_names,
        str(reports_dir / "class_distribution.png"),
        "Test Set Class Distribution"
    )

    # Per-class F1
    plot_per_class_f1(
        results["classification_report"],
        str(reports_dir / "per_class_f1.png"),
        "Per-Class F1 Score — Real-World Test Set"
    )

    # --- Generate report ---
    elapsed = time.time() - start
    report = generate_report(results, errors, label_names, elapsed)
    report_path = reports_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Report] Saved: {report_path}")

    print(f"\nTotal evaluation time: {elapsed:.1f}s")
    print("=" * 60)


def generate_report(results, errors, label_names, elapsed):
    """Generate markdown evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Evaluation Report — Real-World Test Set",
        f"\nGenerated: {now}",
        f"\n## Overall Metrics",
        f"\n| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {results['accuracy']:.4f} |",
        f"| Macro Precision | {results['precision']:.4f} |",
        f"| Macro Recall | {results['recall']:.4f} |",
        f"| Macro F1 | {results['f1']:.4f} |",
        f"| Top-3 Accuracy | {results['top3_accuracy']:.4f} |",
        f"| Total Test Samples | {len(results['labels'])} |",
        f"| Evaluation Time | {elapsed:.1f}s |",
        f"\n## Per-Class Performance",
        f"\n| Class | Precision | Recall | F1 | Support |",
        f"|-------|-----------|--------|-----|---------|",
    ]

    report_dict = results["classification_report"]
    for key, val in report_dict.items():
        if key not in ("accuracy", "macro avg", "weighted avg") and isinstance(val, dict):
            lines.append(
                f"| {key} | {val.get('precision',0):.3f} | "
                f"{val.get('recall',0):.3f} | {val.get('f1-score',0):.3f} | "
                f"{val.get('support',0)} |"
            )

    lines.extend([
        f"\n## Error Analysis",
        f"\n- Total errors: {errors['total_errors']}",
        f"- Error rate: {errors['error_rate']:.4f}",
        f"\n### Top Confusion Pairs",
        f"\n| Actual → Predicted | Count |",
        f"|-------------------|-------|",
    ])

    for pair, count in errors["top_confusion_pairs"][:10]:
        lines.append(f"| {pair} | {count} |")

    lines.extend([
        f"\n## Visualizations",
        f"\n![Confusion Matrix](confusion_matrix.png)",
        f"![Class Distribution](class_distribution.png)",
        f"![Per-Class F1](per_class_f1.png)",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
