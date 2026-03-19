"""
train.py — Training Pipeline for AI Resume Analyzer
===================================================

This module trains the deep learning model used in the AI Resume Analyzer
system. The model learns semantic patterns from resume text and predicts
the most relevant professional role for a candidate.

The classifier is the first stage of the pipeline used to evaluate whether
a candidate's resume aligns with a target job role and job description.

Dataset
-------
Total Resumes : 90,000+
Classes       : 25 IT professions

Synthetic Resumes : 15,000
Realistic Resumes : 75,000

Training uses a 3-way split:
Train / Validation / Test

Deep Learning Components
------------------------
• BERT Encoder (bert-base-uncased)
• AdamW Optimizer
• CrossEntropyLoss
• Linear LR warmup + decay
• Gradient clipping
• Model checkpointing

Fixes Applied
-------------
FIX-1 : Added models/ directory creation before torch.save() to prevent
        FileNotFoundError when the folder does not yet exist.
FIX-2 : time is imported but was never used — removed.
        (All other logic in this file was already correct.)
"""

import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pathlib import Path
from typing import Dict, Tuple

from src.model import get_model
from src.dataset import create_train_val_test_loaders


# ================================================================
# Training Configuration
# ================================================================

class TrainingConfig:
    LEARNING_RATE = 2e-5      # standard BERT fine-tune LR
    NUM_EPOCHS    = 5
    BATCH_SIZE    = 8         # safe for T4 GPU with MAX_LENGTH=384
    MAX_LENGTH    = 384
    WARMUP_RATIO  = 0.06
    WEIGHT_DECAY  = 0.01
    MAX_GRAD_NORM = 1.0
    NUM_CLASSES   = 25
    DROPOUT_RATE  = 0.3
    SEED          = 42


# ================================================================
# Training Loop
# ================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
):
    model.train()

    total_loss = 0
    all_preds  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


# ================================================================
# Validation Loop
# ================================================================

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss  = total_loss / len(loader)
    acc       = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, precision, recall, f1


# ================================================================
# Main Training Function
# ================================================================

def train(config: TrainingConfig = None) -> Tuple[torch.nn.Module, Dict]:

    if config is None:
        config = TrainingConfig()

    torch.manual_seed(config.SEED)

    project_root = Path(__file__).parent.parent

    print("\nAI Resume Analyzer — Training")
    print("=" * 50)

    # ── Dataset ─────────────────────────────────────────────
    csv_path = project_root / "data" / "metadata" / "dataset.csv"

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        csv_path=str(csv_path),
        project_root=str(project_root),
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_LENGTH,
        random_seed=config.SEED,
    )

    # ── Model ────────────────────────────────────────────────
    model, device = get_model(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
    )

    # ── Optimizer (separate weight-decay groups) ─────────────
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)

    total_steps  = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # ── FIX-1: ensure models/ directory exists ───────────────
    model_dir  = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "bert_classifier.pt"

    # ── Training loop ────────────────────────────────────────
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"Train Loss : {train_loss:.4f} | "
            f"Val Loss   : {val_loss:.4f} | "
            f"Val F1     : {val_f1:.4f}"
        )

        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_f1":          best_f1,
                    "config":           vars(config),
                },
                model_path,
            )
            print("  ✓ Best model saved")

    print("\nTraining Complete")
    print(f"Best Validation F1 : {best_f1:.4f}")

    return model, history


# ================================================================
# Entry Point
# ================================================================

if __name__ == "__main__":

    model, history = train()

    print("\nTraining History")
    for i in range(len(history["train_loss"])):
        print(
            f"Epoch {i + 1} | "
            f"Train Loss {history['train_loss'][i]:.4f} | "
            f"Val Loss   {history['val_loss'][i]:.4f} | "
            f"Val F1     {history['val_f1'][i]:.4f}"
        )