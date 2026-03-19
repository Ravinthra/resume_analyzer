"""
ranking.py — Candidate Ranking System
=======================================

Given a job description, ranks resumes by semantic similarity using BERT embeddings.

Features:
  - Batch encoding for efficiency
  - Embedding caching
  - Cosine similarity ranking
  - Top-K candidate retrieval

Author: AI Resume Analyzer Project
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time


class CandidateRanker:
    """
    Ranks candidate resumes against a job description using BERT embeddings.

    Usage:
        ranker = CandidateRanker()
        results = ranker.rank_candidates(jd_text, resume_texts, top_k=10)
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self._cache = {}
        print(f"[Ranker] Loaded {model_name} on {self.device}")

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text into a BERT [CLS] embedding."""
        cache_key = hash(text[:200])  # Use first 200 chars as key
        if cache_key in self._cache:
            return self._cache[cache_key]

        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._cache[cache_key] = embedding
        return embedding

    def batch_encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encode a batch of texts into BERT embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch_texts, add_special_tokens=True, max_length=self.max_length,
                padding="max_length", truncation=True,
                return_attention_mask=True, return_tensors="pt"
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def rank_candidates(
        self,
        jd_text: str,
        resume_texts: List[str],
        resume_labels: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rank candidate resumes against a job description.

        Returns: List of top_k candidates with scores.
        """
        start = time.time()

        # Encode JD
        jd_embedding = self._encode(jd_text)

        # Encode all resumes
        resume_embeddings = self.batch_encode(resume_texts)

        # Compute cosine similarity (embeddings are already normalized)
        similarities = resume_embeddings @ jd_embedding

        # Rank
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            result = {
                "rank": rank + 1,
                "index": int(idx),
                "score": float(similarities[idx]) * 100,  # 0-100 scale
            }
            if resume_labels and idx < len(resume_labels):
                result["label"] = resume_labels[idx]
            results.append(result)

        elapsed = time.time() - start
        print(f"[Ranker] Ranked {len(resume_texts)} candidates in {elapsed:.1f}s")

        return results

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
