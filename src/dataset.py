"""
dataset.py — PyTorch Dataset for BERT Resume Classification (90K Scale)
========================================================================

Supports:
  - 90K+ resumes across 25 IT profession classes
  - Synthetic + real resume sources
  - 3-way train/val/test split
  - Class-organized directory structure (data/<type>/<class>/resume.txt)

Author: AI Resume Analyzer Project
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, Tuple, Optional, List

from src.utils import preprocess_resume


class ResumeDataset(Dataset):
    """
    PyTorch Dataset for resume classification with BERT.

    Supports both legacy (filename column) and new (resume_path column) CSV formats.
    """

    def __init__(
        self,
        csv_path: str,
        project_root: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 256,
        source_filter: Optional[str] = None
    ):
        """
        Args:
            csv_path: Path to dataset CSV
            project_root: Root directory of the project (paths resolved relative to this)
            tokenizer_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length (256 for 90K scale)
            source_filter: Optional — 'synthetic' or 'real' to filter by source
        """
        self.data = pd.read_csv(csv_path)
        self.project_root = project_root
        self.max_length = max_length

        # Support both old format (filename column) and new format (resume_path column)
        if "resume_path" in self.data.columns:
            self.path_column = "resume_path"
        else:
            self.path_column = "filename"

        # Filter by source if requested
        if source_filter and "source" in self.data.columns:
            self.data = self.data[self.data["source"] == source_filter].reset_index(drop=True)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        print(f"[Dataset] Loaded: {len(self.data)} samples (max_len={max_length})")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        # Resolve file path
        rel_path = row[self.path_column]
        if self.path_column == "resume_path":
            file_path = os.path.join(self.project_root, rel_path)
        else:
            # Legacy format: filename relative to data/resumes/
            file_path = os.path.join(self.project_root, "data", "resumes", rel_path)

        text = preprocess_resume(file_path)
        label = int(row["label_id"])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(
    csv_path: str,
    project_root: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 256,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders (2-way split).
    Backward-compatible with old code.
    """
    full_dataset = ResumeDataset(
        csv_path=csv_path,
        project_root=project_root,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"\n[Split] {train_size} train / {val_size} validation")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    print(f"[Batches] Train: {len(train_loader)} | Val: {len(val_loader)}")
    return train_loader, val_loader


def create_train_val_test_loaders(
    csv_path: str,
    project_root: str,
    batch_size: int = 8,
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 256,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create 3-way train/val/test split for 90K pipeline.

    Strategy:
      - Synthetic data → 80% train, 20% val
      - Real data → 67% train, 13% val, 20% test
      - Test set is REAL ONLY (tests generalization)

    Returns: (train_loader, val_loader, test_loader)
    """
    full_data = pd.read_csv(csv_path)
    generator = torch.Generator().manual_seed(random_seed)

    # Split by source
    synth_data = full_data[full_data["source"] == "synthetic"].reset_index(drop=True)
    real_data = full_data[full_data["source"] == "real"].reset_index(drop=True)

    print(f"[Data] Synthetic: {len(synth_data)} | Real: {len(real_data)}")

    # Synthetic: 80% train, 20% val
    synth_train_size = int(len(synth_data) * 0.8)
    synth_val_size = len(synth_data) - synth_train_size

    # Real: 67% train, 13% val, 20% test
    real_test_size = int(len(real_data) * 0.2)
    real_val_size = int(len(real_data) * 0.13)
    real_train_size = len(real_data) - real_test_size - real_val_size

    # Combine train/val CSVs
    synth_shuffled = synth_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    real_shuffled = real_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_df = pd.concat([
        synth_shuffled.iloc[:synth_train_size],
        real_shuffled.iloc[:real_train_size]
    ]).reset_index(drop=True)

    val_df = pd.concat([
        synth_shuffled.iloc[synth_train_size:],
        real_shuffled.iloc[real_train_size:real_train_size + real_val_size]
    ]).reset_index(drop=True)

    test_df = real_shuffled.iloc[real_train_size + real_val_size:].reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} (real only)")

    # Helper to create dataset from DataFrame
    def df_to_loader(df, shuffle):
        ds = _DataFrameDataset(df, project_root, tokenizer_name, max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = df_to_loader(train_df, shuffle=True)
    val_loader = df_to_loader(val_df, shuffle=False)
    test_loader = df_to_loader(test_df, shuffle=False)

    print(f"[Batches] Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


class _DataFrameDataset(Dataset):
    """Internal dataset that works directly from a DataFrame (no CSV reload)."""

    def __init__(self, df, project_root, tokenizer_name, max_length):
        self.data = df
        self.project_root = project_root
        self.max_length = max_length
        self.path_column = "resume_path" if "resume_path" in df.columns else "filename"
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rel_path = row[self.path_column]
        if self.path_column == "resume_path":
            file_path = os.path.join(self.project_root, rel_path)
        else:
            file_path = os.path.join(self.project_root, "data", "resumes", rel_path)

        text = preprocess_resume(file_path)
        label = int(row["label_id"])

        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent

    # Test with new 90K metadata
    meta_csv = project_root / "data" / "metadata" / "dataset.csv"
    if meta_csv.exists():
        print("=" * 60)
        print("90K DATASET TEST")
        print("=" * 60)
        train_loader, val_loader, test_loader = create_train_val_test_loaders(
            csv_path=str(meta_csv),
            project_root=str(project_root),
            batch_size=8,
            max_length=128  # Short for quick test
        )
        batch = next(iter(train_loader))
        print(f"\n[Batch] input_ids: {batch['input_ids'].shape} | labels: {batch['label'].tolist()}")
    else:
        # Fallback to legacy format
        csv_path = str(project_root / "data" / "dataset.csv")
        print("=" * 60)
        print("LEGACY DATASET TEST")
        print("=" * 60)
        train_loader, val_loader = create_data_loaders(
            csv_path=csv_path,
            project_root=str(project_root),
            batch_size=4,
            max_length=128
        )
