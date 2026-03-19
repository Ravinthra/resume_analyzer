"""
evaluation.py — Evaluation Engine for Resume Classifier & Matching System
==========================================================================

Metrics:
  - Classifier: Accuracy, Precision, Recall, F1, Confusion Matrix
  - Job Matching: Mean cosine similarity, Top-K accuracy, ranking quality
  - Skill Extraction: Precision, recall of extracted skills
  - Error Analysis: Misclassified samples, confusion pairs

Author: AI Resume Analyzer Project
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


# ===========================================================================
# CLASSIFIER EVALUATION
# ===========================================================================

def evaluate_classifier(
    model,
    test_loader: DataLoader,
    device: torch.device,
    label_names: List[str]
) -> Dict:
    """
    Evaluate the BERT classifier on the test set.

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix,
        per_class_report, all_predictions, all_labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_logits.extend(logits.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class report
    unique_labels = sorted(set(all_labels))
    target_names = [label_names[i] for i in unique_labels if i < len(label_names)]
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Top-3 accuracy
    logits_np = np.array(all_logits)
    top3_correct = 0
    for i, label in enumerate(all_labels):
        top3_classes = np.argsort(logits_np[i])[-3:]
        if label in top3_classes:
            top3_correct += 1
    top3_accuracy = top3_correct / len(all_labels) if all_labels else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top3_accuracy": top3_accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "logits": logits_np,
    }


# ===========================================================================
# JOB MATCHING EVALUATION
# ===========================================================================

def evaluate_job_matching(
    analyzer,
    test_cases: List[Dict]
) -> Dict:
    """
    Evaluate resume ↔ job description matching quality.

    test_cases: List of {resume_text, jd_text, expected_match (bool)}

    Returns: Dict with avg scores, ranking quality metrics
    """
    correct_scores = []
    incorrect_scores = []

    for case in test_cases:
        score = analyzer.compute_similarity(case["resume_text"], case["jd_text"])
        if case.get("expected_match", False):
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)

    avg_correct = np.mean(correct_scores) if correct_scores else 0
    avg_incorrect = np.mean(incorrect_scores) if incorrect_scores else 0
    separation = avg_correct - avg_incorrect

    return {
        "avg_correct_match_score": float(avg_correct),
        "avg_incorrect_match_score": float(avg_incorrect),
        "score_separation": float(separation),
        "num_correct_cases": len(correct_scores),
        "num_incorrect_cases": len(incorrect_scores),
    }


# ===========================================================================
# SKILL EXTRACTION EVALUATION
# ===========================================================================

def evaluate_skill_extraction(
    test_cases: List[Dict]
) -> Dict:
    """
    Evaluate skill extraction accuracy.

    test_cases: List of {text, expected_skills: set}

    Returns: Dict with avg precision, recall, samples
    """
    from src.utils import extract_skills

    precisions = []
    recalls = []
    samples = []

    for case in test_cases:
        expected = set(s.lower() for s in case["expected_skills"])
        extracted = set(s.lower() for s in extract_skills(case["text"]))

        if extracted:
            precision = len(expected & extracted) / len(extracted)
        else:
            precision = 0.0

        if expected:
            recall = len(expected & extracted) / len(expected)
        else:
            recall = 1.0

        precisions.append(precision)
        recalls.append(recall)
        samples.append({
            "expected": list(expected),
            "extracted": list(extracted),
            "precision": precision,
            "recall": recall,
        })

    return {
        "avg_precision": float(np.mean(precisions)) if precisions else 0,
        "avg_recall": float(np.mean(recalls)) if recalls else 0,
        "num_samples": len(samples),
        "samples": samples[:10],  # Return first 10 for inspection
    }


# ===========================================================================
# ERROR ANALYSIS
# ===========================================================================

def analyze_errors(
    predictions: List[int],
    labels: List[int],
    label_names: List[str],
    logits: Optional[np.ndarray] = None
) -> Dict:
    """
    Analyze misclassifications.

    Returns: Dict with error rate per class, confusion pairs,
             low-confidence predictions, misclassified samples
    """
    errors = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            entry = {
                "index": i,
                "predicted": label_names[pred] if pred < len(label_names) else str(pred),
                "actual": label_names[label] if label < len(label_names) else str(label),
            }
            if logits is not None:
                probs = torch.softmax(torch.tensor(logits[i]), dim=0).numpy()
                entry["confidence"] = float(probs[pred])
                entry["true_class_prob"] = float(probs[label])
            errors.append(entry)

    # Error rate per class
    error_rate = {}
    for label_id in set(labels):
        name = label_names[label_id] if label_id < len(label_names) else str(label_id)
        class_indices = [i for i, l in enumerate(labels) if l == label_id]
        class_errors = [i for i in class_indices if predictions[i] != labels[i]]
        error_rate[name] = len(class_errors) / len(class_indices) if class_indices else 0

    # Confusion pairs (most common misclassifications)
    confusion_pairs = {}
    for e in errors:
        pair = f"{e['actual']} → {e['predicted']}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # Low confidence predictions
    low_conf = []
    if logits is not None:
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        for i in range(len(predictions)):
            max_prob = float(np.max(probs[i]))
            if max_prob < 0.5:
                low_conf.append({
                    "index": i,
                    "predicted": label_names[predictions[i]] if predictions[i] < len(label_names) else str(predictions[i]),
                    "actual": label_names[labels[i]] if labels[i] < len(label_names) else str(labels[i]),
                    "confidence": max_prob,
                })

    return {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(labels) if labels else 0,
        "error_rate_per_class": error_rate,
        "top_confusion_pairs": sorted_pairs[:15],
        "low_confidence_predictions": low_conf[:20],
        "misclassified_samples": errors[:20],
    }
