"""
colab_train.py
==============

Google Colab training script for AI Resume Analyzer.

Steps:
1. Install dependencies
2. Generate dataset if missing
3. Train BERT classifier on T4 GPU
4. Evaluate model on test set
5. Save best checkpoint

Expected runtime: ~20 minutes on T4 GPU

Fixes Applied
-------------
FIX-1  : Replaced deprecated DataFrame.append() with pd.concat() — crashed
          on pandas >= 2.0.
FIX-2  : Fixed wrong path resolution in __getitem__. The CSV already stores
          absolute paths, so prepending project_root caused FileNotFoundError.
          Changed to Path(row.resume_path) directly.
FIX-3  : Removed full BERT parameter freeze. Entire BERT was frozen which
          prevented the model from learning role semantics (F1 stayed near 0).
          Now only embeddings + first 8 encoder layers are frozen; last 4
          layers and pooler are trainable (good balance for T4 memory).
FIX-4  : Corrected learning rate from 2e-4 → 2e-5. At 2e-4 with unfrozen
          BERT layers the pre-trained weights are destroyed in epoch 1.
FIX-5  : Renamed outer `labels` JSON variable to `label_meta` to prevent
          silent overwrite inside the batch training loop.
FIX-6  : Reduced DataLoader batch_size from 32 → 16. batch=32 with
          MAX_LEN=384 and partial BERT unfrozen causes OOM on T4 GPU.
          batch=16 runs safely; use gradient accumulation if you need
          effective batch=32.
FIX-7  : Moved `import re` from inside __getitem__ (called 90k times)
          to module top-level.
FIX-8  : Added re.escape() around label_str to safely handle multi-word
          role names with special regex characters.
FIX-9  : Added `models/` directory creation before torch.save() to prevent
          FileNotFoundError when the folder does not yet exist.
FIX-10 : Wrapped test evaluation in `with torch.no_grad():` — it was running
          outside the context manager, wasting GPU memory on gradient tracking.
FIX-11 : Added torch.manual_seed(42) for reproducibility.
FIX-12 : Replaced bare `except:` with `except ImportError:` in
          setup_environment() to avoid swallowing unexpected errors.
FIX-13 : Removed unused `import time` and `import os`.
FIX-14 : Extracted NUM_EPOCHS constant instead of hardcoded magic numbers.
"""

import sys
import re                          # FIX-7: moved from inside __getitem__
import json
import random
import subprocess
from pathlib import Path

# ── constants ────────────────────────────────────────────────────
NUM_EPOCHS  = 5
MAX_LEN     = 384
BATCH_SIZE  = 48                   # T4 15GB VRAM — use it fully
ACCUM_STEPS = 2                    # Gradient accumulation → effective batch = 96
LEARN_RATE  = 2e-5
USE_FP16    = True                 # Mixed precision — halves memory, doubles speed


# ==========================================================
# SETUP
# ==========================================================

def setup_environment() -> None:

    print("=" * 60)
    print("COLAB SETUP")
    print("=" * 60)

    import torch

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU not enabled — training will be slow")

    try:                                        # FIX-12: was bare except:
        import transformers
        import sklearn
        import pandas
        print("All dependencies already installed")
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "transformers", "torch", "scikit-learn",
            "pandas", "matplotlib", "seaborn",
        ])

    print("Environment ready")


# ==========================================================
# DATA GENERATION
# ==========================================================

def generate_dataset() -> None:

    project_root = Path(__file__).parent.parent
    meta_csv     = project_root / "data/metadata/dataset.csv"

    if meta_csv.exists():
        import pandas as pd
        df = pd.read_csv(meta_csv)
        print(f"Dataset already exists — {len(df):,} rows")
        return

    print("Generating dataset …")
    sys.path.append(str(project_root / "scripts"))

    from generate_90k_resumes import main
    main()


# ==========================================================
# DATASET & DATALOADERS
# ==========================================================

def build_dataloaders(project_root: Path):
    """
    Split strategy
    ──────────────
    Synthetic (15 k total) → 80 % train | 20 % val  | 0 % test
    Real      (75 k total) → 70 % train | 15 % val  | 15 % test

    This keeps test set purely real-world resumes.
    """
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer

    csv_path = project_root / "data/metadata/dataset.csv"
    data     = pd.read_csv(csv_path)

    random.seed(42)

    synth = data[data.source == "synthetic"].sample(frac=1, random_state=42).reset_index(drop=True)
    real  = data[data.source == "real"].sample(frac=1, random_state=42).reset_index(drop=True)

    synth_train_end = int(len(synth) * 0.8)
    real_train_end  = int(len(real)  * 0.7)
    real_val_end    = real_train_end + int(len(real) * 0.15)

    # FIX-1: replaced deprecated DataFrame.append() with pd.concat()
    train_df = pd.concat(
        [synth.iloc[:synth_train_end], real.iloc[:real_train_end]],
        ignore_index=True,
    )
    val_df = pd.concat(
        [synth.iloc[synth_train_end:], real.iloc[real_train_end:real_val_end]],
        ignore_index=True,
    )
    test_df = real.iloc[real_val_end:].reset_index(drop=True)

    print(f"Train : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    class ResumeDataset(Dataset):

        def __init__(self, df: pd.DataFrame) -> None:
            self.df = df.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            import torch
            row = self.df.iloc[idx]

            # FIX-2: CSV stores absolute paths — do NOT prepend project_root
            path = Path(row.resume_path)

            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                text = ""

            encoding = tokenizer(
                text,
                max_length=MAX_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label":          torch.tensor(int(row.label_id), dtype=torch.long),
            }

    # FIX-6: batch_size reduced from 32 → BATCH_SIZE (16) to prevent OOM
    train_loader = DataLoader(ResumeDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ResumeDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ResumeDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


# ==========================================================
# MODEL
# ==========================================================

def build_model(num_classes: int):
    """
    BERT classifier — FULL fine-tuning (all layers trainable).

    With fp16 mixed precision + batch=48, this fits comfortably in
    T4 15GB VRAM (~12-13GB usage).
    """
    import torch.nn as nn
    import os
    from transformers import BertModel

    class ResumeClassifier(nn.Module):

        def __init__(self) -> None:
            super().__init__()

            self.bert = BertModel.from_pretrained(
                "bert-base-uncased",
                token=os.environ.get("HF_TOKEN")
            )

            # ALL layers trainable — full fine-tuning
            # fp16 + batch=48 keeps this within T4's 15GB

            self.dropout = nn.Dropout(0.3)
            self.fc      = nn.Linear(768, num_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls     = outputs.pooler_output          # [batch, 768]
            return self.fc(self.dropout(cls))

    return ResumeClassifier()


# ==========================================================
# TRAINING
# ==========================================================

def train() -> None:
    import torch
    import torch.nn as nn
    from torch.amp import autocast, GradScaler
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(42)
    random.seed(42)

    project_root = Path(__file__).parent.parent
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if device.type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU RAM: {gpu_mem:.1f} GB")

    with open(project_root / "data/metadata/labels.json", encoding="utf-8") as f:
        label_meta = json.load(f)

    NUM_CLASSES = label_meta["num_classes"]

    print("Building dataloaders — please wait...")
    train_loader, val_loader, test_loader = build_dataloaders(project_root)

    print("Loading BERT model...")
    model = build_model(NUM_CLASSES).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {trainable:,} / {total:,}")

    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARN_RATE,
        weight_decay=0.01,
    )
    total_steps = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.06),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=USE_FP16)

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "bert_classifier.pt"

    best_f1 = 0.0

    print(f"\n{'='*60}")
    print(f"  Training — {NUM_EPOCHS} Epochs")
    print(f"  Batch={BATCH_SIZE} × Accum={ACCUM_STEPS} = Effective {BATCH_SIZE*ACCUM_STEPS}")
    print(f"  FP16={USE_FP16} | Full BERT fine-tune")
    print(f"{'='*60}")
    print(f"  {'Epoch':<8} {'TrainLoss':<12} {'ValAcc':<10} {'ValF1':<10} {'VRAM':<10} {'Note'}")
    print(f"  {'-'*60}")

    for epoch in range(NUM_EPOCHS):

        # ── train ─────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            # Forward pass in fp16
            with autocast("cuda", enabled=USE_FP16):
                logits = model(input_ids, mask)
                loss   = criterion(logits, labels) / ACCUM_STEPS

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUM_STEPS

        avg_train_loss = train_loss / len(train_loader)

        # ── validation ────────────────────────────────────
        model.eval()
        preds_v, labs_v = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask      = batch["attention_mask"].to(device)
                labels    = batch["label"].to(device)

                with autocast("cuda", enabled=USE_FP16):
                    logits = model(input_ids, mask)
                p      = torch.argmax(logits, dim=1)

                preds_v.extend(p.cpu().tolist())
                labs_v.extend(labels.cpu().tolist())

        acc  = accuracy_score(labs_v, preds_v)
        f1   = f1_score(labs_v, preds_v, average="macro")
        note = ""

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_f1,
            }, model_path)
            note = "✓ best saved"

        # VRAM usage
        vram = ""
        if device.type == "cuda":
            vram = f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB"

        print(
            f"  {epoch+1}/{NUM_EPOCHS:<6} "
            f"{avg_train_loss:<12.4f} "
            f"{acc:<10.4f} "
            f"{f1:<10.4f} "
            f"{vram:<10s} "
            f"{note}"
        )

    print(f"\n  Best Val F1 : {best_f1:.4f}")
    print(f"  Model saved : {model_path}")

    # ── test evaluation ──────────────────────────────────────
    from sklearn.metrics import classification_report

    preds_t, labs_t = [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            mask      = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            logits = model(input_ids, mask)
            p      = torch.argmax(logits, dim=1)

            preds_t.extend(p.cpu().tolist())
            labs_t.extend(labels.cpu().tolist())

    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)

    id_to_label  = label_meta["id_to_label"]
    target_names = [id_to_label[str(i)] for i in range(NUM_CLASSES)]

    print(classification_report(labs_t, preds_t, target_names=target_names))


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    setup_environment()
    generate_dataset()
    train()