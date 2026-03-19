"""
predict.py -- Inference Pipeline for AI Resume Analyzer
========================================================

This module implements the complete prediction pipeline:
1. Load trained BERT model from checkpoint
2. Classify resume into job roles (Stage 10)
3. Match resume against job descriptions using cosine similarity (Stage 8)
4. Detect missing skills via set difference (Stage 9)
5. Return a comprehensive analysis result

Deep Learning Concepts Demonstrated:
- Model loading from checkpoint (state_dict)
- Inference mode (no gradients, no dropout)
- Embedding extraction from BERT
- Cosine similarity for semantic matching
- Softmax for probability conversion

Author: AI Resume Analyzer Project
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import BertTokenizer, BertModel

from src.model import ResumeClassifier
from src.utils import (
    preprocess_resume,
    extract_skills,
    extract_skills_from_jd,
    find_missing_skills,
    get_skill_match_percentage,
    identify_resume_strengths,
    generate_improvement_suggestions,
    load_text_file,
    load_labels,
    clean_text,
    truncate_for_bert
)


class ResumeAnalyzer:
    """
    Complete inference pipeline for resume analysis.
    
    Combines three capabilities:
    A. Role Classification -- Predict the candidate's job role
    B. Job Matching        -- Score resume against job descriptions
    C. Skill Gap Analysis  -- Find missing skills
    
    Usage:
        analyzer = ResumeAnalyzer()
        result = analyzer.analyze("path/to/resume.pdf")
    """
    
    def __init__(self, model_path: str = None, labels_path: str = None,
                 jd_dir: str = None):
        """
        Initialize the analyzer by loading the trained model and resources.
        
        Args:
            model_path: Path to saved model checkpoint (.pt)
            labels_path: Path to labels.json
            jd_dir: Directory containing job descriptions
        """
        self.project_root = Path(__file__).parent.parent
        
        # Set default paths
        if model_path is None:
            model_path = str(self.project_root / "models" / "bert_classifier.pt")
        if labels_path is None:
            labels_path = str(self.project_root / "data" / "labels.json")
        if jd_dir is None:
            jd_dir = str(self.project_root / "data" / "job_descriptions")
        
        self.jd_dir = jd_dir
        
        # ============================================================
        # Step 1: Set device (CPU/GPU)
        # ============================================================
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[Analyzer] Device: {self.device}")
        
        # ============================================================
        # Step 2: Load label mapping
        # ============================================================
        self.labels = load_labels(labels_path)
        self.id_to_label = self.labels["id_to_label"]
        self.num_classes = self.labels["num_classes"]
        print(f"[Analyzer] Labels: {self.num_classes} classes")
        
        # ============================================================
        # Step 3: Load trained classification model
        # ============================================================
        # torch.load() restores the saved checkpoint dictionary
        # map_location ensures it loads on the correct device
        # (e.g., model trained on GPU but loaded on CPU)
        self.classifier = self._load_classifier(model_path)
        
        # ============================================================
        # Step 4: Load BERT model for embeddings (similarity matching)
        # ============================================================
        # We use a SEPARATE BertModel instance for embedding extraction
        # This is the pretrained BERT without our classification head
        # Why separate? The classifier's BERT has been fine-tuned for
        # classification, but for similarity we want general embeddings
        self.embedding_model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding_model = self.embedding_model.to(self.device)
        self.embedding_model.eval()  # Always in eval mode for inference
        
        # ============================================================
        # Step 5: Initialize tokenizer
        # ============================================================
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # ============================================================
        # Step 6: Pre-load job descriptions and their embeddings
        # ============================================================
        self.job_descriptions = self._load_job_descriptions()
        self.jd_embeddings = self._compute_jd_embeddings()
        
        print(f"[Analyzer] Loaded {len(self.job_descriptions)} job descriptions")
        print("[Analyzer] Ready for analysis!")
    
    def _load_classifier(self, model_path: str) -> ResumeClassifier:
        """
        Load the trained classifier from a checkpoint file.
        
        Why we load state_dict instead of the full model:
        - state_dict contains only learned parameters (weights + biases)
        - It's decoupled from the model class definition
        - Allows loading even if the code has changed (as long as architecture matches)
        - More portable across PyTorch versions
        """
        # Create a new model instance with the same architecture
        model = ResumeClassifier(
            num_classes=self.num_classes,
            dropout_rate=0.3  # Must match training config
        )
        
        if os.path.exists(model_path):
            # Load checkpoint
            # weights_only=False needed for complex checkpoint dicts
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=False)
            
            # Handle both formats: dict with model_state_dict key, or raw state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                best_f1 = checkpoint.get('best_val_f1', 'N/A')
            else:
                state_dict = checkpoint
                best_f1 = 'N/A'
            
            model.load_state_dict(state_dict, strict=False)
            print(f"[Analyzer] Model loaded from {model_path}")
            print(f"[Analyzer] Best F1: {best_f1}")
        else:
            print(f"[Analyzer] WARNING: No checkpoint found at {model_path}")
            print("[Analyzer] Using untrained model (results will be random)")
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode (disables dropout)
        
        return model
    
    def _tokenize(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """
        Tokenize text for BERT input.
        
        Returns dictionary with input_ids and attention_mask tensors
        ready to be fed into the model.
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device)
        }
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Extract BERT [CLS] embedding for a piece of text.
        
        The [CLS] token embedding serves as a fixed-size representation
        of the entire input sequence. It captures the semantic meaning
        in a 768-dimensional vector.
        
        How it works:
        1. Tokenize text -> input_ids, attention_mask
        2. Pass through BERT -> all token embeddings
        3. Extract [CLS] token (first token) embedding
        4. Return 768-dim vector
        
        Returns:
            Tensor of shape [768] (normalized)
        """
        tokens = self._tokenize(text)
        
        with torch.no_grad():
            output = self.embedding_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )
        
        # Use pooler_output: [CLS] token embedding (after pooling layer)
        # Shape: [1, 768] -> squeeze to [768]
        embedding = output.pooler_output.squeeze(0)
        
        # Normalize the embedding to unit length
        # This makes cosine similarity equivalent to dot product
        # and ensures all embeddings are on the same scale
        embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding
    
    def _load_job_descriptions(self) -> Dict[str, Dict]:
        """
        Load job descriptions from the JD directory.
        
        Handles both flat layout (jd_data_scientist.txt in root)
        and subdirectory layout (data_scientist/jd_0000.txt).
        Loads one representative JD per class.
        
        Returns:
            Dictionary mapping JD name to {text, skills}
        """
        jds = {}
        
        if not os.path.exists(self.jd_dir):
            print(f"[Analyzer] JD directory not found: {self.jd_dir}")
            return jds
        
        for entry in os.listdir(self.jd_dir):
            entry_path = os.path.join(self.jd_dir, entry)
            
            if os.path.isdir(entry_path):
                # Subdirectory layout: data_scientist/jd_0000.txt
                # Load the first JD file from each subdirectory
                name = entry.replace("_", " ").title()
                for fname in sorted(os.listdir(entry_path)):
                    if fname.endswith(".txt"):
                        filepath = os.path.join(entry_path, fname)
                        text = load_text_file(filepath)
                        jds[name] = {
                            "text": text,
                            "cleaned_text": clean_text(text),
                            "skills": extract_skills(text),
                            "filename": fname
                        }
                        break  # One JD per class is enough
            
            elif entry.endswith(".txt"):
                # Flat layout: jd_data_scientist.txt
                filepath = entry_path
                text = load_text_file(filepath)
                name = entry.replace("jd_", "").replace(".txt", "")
                name = name.replace("_", " ").title()
                jds[name] = {
                    "text": text,
                    "cleaned_text": clean_text(text),
                    "skills": extract_skills(text),
                    "filename": entry
                }
        
        return jds
    
    def _compute_jd_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Pre-compute BERT embeddings for all job descriptions.
        
        We compute these once at initialization because:
        - JDs don't change between requests
        - Embedding computation is expensive (BERT forward pass)
        - Caching avoids redundant computation for each resume
        """
        embeddings = {}
        
        for name, jd_data in self.job_descriptions.items():
            preprocessed = truncate_for_bert(jd_data["cleaned_text"])
            embeddings[name] = self._get_embedding(preprocessed)
        
        return embeddings
    
    # ================================================================
    # PIPELINE A: Job Role Classification
    # ================================================================
    
    def classify_role(self, text: str) -> Dict:
        """
        Predict the most likely job role for a resume.
        
        Pipeline:
        1. Tokenize preprocessed text
        2. Forward pass through classifier
        3. Apply softmax to get probabilities
        4. Select top prediction
        
        Args:
            text: Preprocessed resume text
            
        Returns:
            Dictionary with predicted role and confidence scores
        """
        tokens = self._tokenize(text)
        
        # Forward pass (no gradients needed for inference)
        with torch.no_grad():
            logits = self.classifier(
                tokens["input_ids"],
                tokens["attention_mask"]
            )
        
        # ============================================================
        # Convert logits to probabilities using Softmax
        # ============================================================
        # Softmax formula: P(class_i) = exp(logit_i) / sum(exp(logit_j))
        #
        # This transforms raw scores (logits) into a probability
        # distribution that sums to 1.0
        #
        # Example:
        #   logits = [2.0, 1.0, 0.1, -0.5, 0.3]
        #   softmax = [0.50, 0.18, 0.07, 0.04, 0.09]
        #   sum = 1.0
        #
        # dim=1 means softmax is applied across classes (not batch)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get the top prediction
        confidence, predicted_id = torch.max(probabilities, dim=1)
        predicted_label = self.id_to_label[str(predicted_id.item())]
        
        # Get all class probabilities for detailed output
        all_probs = {
            self.id_to_label[str(i)]: round(probabilities[0][i].item() * 100, 2)
            for i in range(self.num_classes)
        }
        
        return {
            "predicted_role": predicted_label,
            "confidence": round(confidence.item() * 100, 2),
            "all_probabilities": all_probs
        }
    
    # ================================================================
    # PIPELINE B: Resume-JD Similarity Matching
    # ================================================================
    
    def match_jobs(self, text: str) -> Dict:
        """
        Match a resume against all job descriptions using cosine similarity.
        
        Cosine Similarity measures the angle between two vectors:
        
            cos(theta) = (A . B) / (|A| * |B|)
        
        - Range: [-1, 1] (after normalization, effectively [0, 1])
        - 1.0 = identical direction (same meaning)
        - 0.0 = perpendicular (unrelated)
        - -1.0 = opposite direction (opposite meaning)
        
        Because we normalize embeddings to unit length in _get_embedding(),
        cosine similarity simplifies to just the dot product: A . B
        
        Why cosine similarity over Euclidean distance:
        - Invariant to vector magnitude (document length doesn't matter)
        - A short resume and a long resume about the same topic
          will have similar direction even if different magnitudes
        
        Args:
            text: Preprocessed resume text
            
        Returns:
            Dictionary with match scores and rankings
        """
        # Get resume embedding
        resume_embedding = self._get_embedding(text)
        
        matches = []
        
        for jd_name, jd_embedding in self.jd_embeddings.items():
            # Compute cosine similarity
            # Since both vectors are normalized, this is just the dot product
            similarity = torch.dot(resume_embedding, jd_embedding).item()
            
            # Convert to percentage (0-100)
            # Shift from [-1,1] to [0,1] then multiply by 100
            match_percentage = max(0, similarity) * 100
            
            matches.append({
                "job_title": jd_name,
                "match_score": round(match_percentage, 2),
                "cosine_similarity": round(similarity, 4)
            })
        
        # Sort by match score (highest first)
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "top_match": matches[0] if matches else None,
            "all_matches": matches
        }
    
    # ================================================================
    # PIPELINE C: Skill Gap Analysis
    # ================================================================
    
    def analyze_skills(self, resume_text: str, 
                       target_jd: str = None) -> Dict:
        """
        Extract skills from resume and identify gaps.
        
        If target_jd is specified, compare against that specific JD.
        Otherwise, compare against the best-matching JD from Pipeline B.
        
        Args:
            resume_text: Raw resume text
            target_jd: Optional specific JD name to compare against
            
        Returns:
            Dictionary with skill analysis
        """
        # Extract skills from resume
        resume_skills = extract_skills(resume_text)
        
        # Determine target JD
        if target_jd and target_jd in self.job_descriptions:
            jd_data = self.job_descriptions[target_jd]
        else:
            # Use the first JD as default (caller should set based on top match)
            jd_name = list(self.job_descriptions.keys())[0] if self.job_descriptions else None
            jd_data = self.job_descriptions.get(jd_name, {"skills": set()})
        
        jd_skills = jd_data.get("skills", set())
        
        # Skill gap analysis
        missing = find_missing_skills(resume_skills, jd_skills)
        match_pct = get_skill_match_percentage(resume_skills, jd_skills)
        common = resume_skills.intersection(jd_skills)
        extra = resume_skills - jd_skills  # Skills in resume but not in JD
        
        return {
            "resume_skills": sorted(resume_skills),
            "resume_skill_count": len(resume_skills),
            "jd_skills": sorted(jd_skills),
            "matching_skills": sorted(common),
            "missing_skills": sorted(missing),
            "extra_skills": sorted(extra),
            "skill_match_percentage": round(match_pct, 2)
        }
    
    # ================================================================
    # MAIN ANALYSIS PIPELINE (Combines A + B + C)
    # ================================================================
    
    def analyze(self, resume_path: str) -> Dict:
        """
        Complete resume analysis pipeline.
        
        This is the main entry point that orchestrates all three pipelines:
        A. Role Classification -> predicted job role + confidence
        B. Job Matching -> ranked list of matching JDs
        C. Skill Analysis -> skills found, missing, and match percentage
        
        Args:
            resume_path: Path to resume file (PDF or text)
            
        Returns:
            Comprehensive analysis dictionary with all results
        """
        print(f"\n{'=' * 60}")
        print(f"ANALYZING RESUME: {os.path.basename(resume_path)}")
        print(f"{'=' * 60}")
        
        # Step 1: Preprocess the resume
        print("\n[1/4] Preprocessing...")
        processed_text = preprocess_resume(resume_path)
        raw_text = load_text_file(resume_path) if not resume_path.endswith('.pdf') else processed_text
        
        # Step 2: Classify role
        print("[2/4] Classifying role...")
        classification = self.classify_role(processed_text)
        
        # Step 3: Match against job descriptions
        print("[3/4] Matching jobs...")
        matching = self.match_jobs(processed_text)
        
        # Step 4: Skill gap analysis (against top matching JD)
        print("[4/4] Analyzing skills...")
        top_jd_name = matching["top_match"]["job_title"] if matching["top_match"] else None
        skills = self.analyze_skills(raw_text, target_jd=top_jd_name)
        
        # Combine results
        result = {
            "resume_file": os.path.basename(resume_path),
            "classification": classification,
            "job_matching": matching,
            "skill_analysis": skills,
        }
        
        # Print summary
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: Dict):
        """Print a formatted analysis result."""
        print(f"\n{'=' * 60}")
        print("ANALYSIS RESULTS")
        print(f"{'=' * 60}")
        
        # Classification
        cls = result["classification"]
        print(f"\n[PREDICTED ROLE] {cls['predicted_role']} "
              f"({cls['confidence']}% confidence)")
        print(f"\n   All probabilities:")
        for role, prob in cls["all_probabilities"].items():
            bar = "#" * int(prob / 2)
            print(f"      {role:<20} {prob:>6.2f}% |{bar}")
        
        # Job matching
        match = result["job_matching"]
        print(f"\n[JOB MATCHING]")
        for m in match["all_matches"]:
            bar = "#" * int(m["match_score"] / 2)
            print(f"      {m['job_title']:<20} {m['match_score']:>6.2f}% |{bar}")
        
        # Skills
        skills = result["skill_analysis"]
        print(f"\n[SKILL ANALYSIS]")
        print(f"   Skills found:   {skills['resume_skill_count']}")
        print(f"   Skill match:    {skills['skill_match_percentage']}%")
        print(f"   Missing skills: {', '.join(skills['missing_skills'][:10])}")
        if len(skills['missing_skills']) > 10:
            print(f"                   ... and {len(skills['missing_skills']) - 10} more")

class ResumeJobAnalyzer:
    """
    Resume vs Job Description Analyzer.
    
    Analyzes a resume against a SPECIFIC job description and role.
    Unlike ResumeAnalyzer (which classifies and matches against stored JDs),
    this class takes user-provided JD text and produces a detailed comparison.
    
    Output format:
    {
        "job_role": "Machine Learning Engineer",
        "match_score": 82,
        "matching_skills": ["Python", "TensorFlow"],
        "missing_skills": ["Kubernetes", "Spark"],
        "resume_strengths": ["Strong ML experience"],
        "improvement_suggestions": ["Add distributed computing experience"]
    }
    """
    
    def __init__(self, model_path: str = None, labels_path: str = None):
        """
        Initialize with BERT model for embeddings and classification.
        
        Args:
            model_path: Path to trained classifier checkpoint
            labels_path: Path to labels.json
        """
        self.project_root = Path(__file__).parent.parent
        
        if model_path is None:
            model_path = str(self.project_root / "models" / "bert_classifier.pt")
        if labels_path is None:
            labels_path = str(self.project_root / "data" / "labels.json")
        
        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[ResumeJobAnalyzer] Device: {self.device}")
        
        # Load label mapping
        self.labels = load_labels(labels_path)
        self.id_to_label = self.labels["id_to_label"]
        self.num_classes = self.labels["num_classes"]
        
        # Load trained classifier (for optional role prediction)
        self._classifier = None
        self._model_path = model_path
        
        # Load BERT model for embedding extraction
        self.embedding_model = BertModel.from_pretrained("bert-base-uncased")
        self.embedding_model = self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        print("[ResumeJobAnalyzer] Ready!")
    
    @property
    def classifier(self):
        """Lazy-load classifier only when needed."""
        if self._classifier is None:
            model = ResumeClassifier(
                num_classes=self.num_classes,
                dropout_rate=0.3
            )
            if os.path.exists(self._model_path):
                checkpoint = torch.load(
                    self._model_path,
                    map_location=self.device,
                    weights_only=False
                )
                # Handle both formats
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()
            self._classifier = model
        return self._classifier
    
    def _tokenize(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """Tokenize text for BERT input."""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device)
        }
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Extract BERT [CLS] embedding for text.
        
        Uses the [CLS] token (first token) of BERT's output as a 
        fixed-size 768-dim representation of the input text.
        Normalized to unit length for cosine similarity.
        """
        tokens = self._tokenize(text)
        
        with torch.no_grad():
            output = self.embedding_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )
        
        # [CLS] token embedding from last hidden state
        # Shape: [1, seq_len, 768] -> [1, 768] -> [768]
        embedding = output.last_hidden_state[:, 0, :].squeeze(0)
        
        # Normalize to unit length for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding
    
    def _classify_role(self, text: str) -> Dict:
        """Predict the most likely job role for a resume."""
        tokens = self._tokenize(text)
        
        with torch.no_grad():
            logits = self.classifier(
                tokens["input_ids"],
                tokens["attention_mask"]
            )
        
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, dim=1)
        predicted_label = self.id_to_label[str(predicted_id.item())]
        
        return {
            "predicted_role": predicted_label,
            "confidence": round(confidence.item() * 100, 2)
        }
    
    def analyze_against_job(self, resume_path: str,
                             job_description: str,
                             job_role: str) -> Dict:
        """
        Analyze a resume against a specific job description.
        
        This is the main entry point. It produces a structured report
        showing how well a resume matches a given job.
        
        Pipeline:
        1. Extract & preprocess resume text
        2. Encode resume with BERT -> 768-dim embedding
        3. Encode JD with BERT -> 768-dim embedding
        4. Cosine similarity -> match score (0-100%)
        5. Extract skills from both resume and JD
        6. Set intersection -> matching skills
        7. Set difference -> missing skills
        8. Generate strengths and improvement suggestions
        
        Args:
            resume_path: Path to resume file (PDF or TXT)
            job_description: Full job description text
            job_role: Target job role (e.g., "ML Engineer")
            
        Returns:
            Structured analysis dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"RESUME vs JD ANALYSIS")
        print(f"Target Role: {job_role}")
        print(f"{'=' * 60}")
        
        # ============================================================
        # Step 1: Preprocess resume
        # ============================================================
        print("\n[1/6] Preprocessing resume...")
        processed_text = preprocess_resume(resume_path)
        raw_text = (
            load_text_file(resume_path)
            if not resume_path.lower().endswith('.pdf')
            else processed_text
        )
        
        # ============================================================
        # Step 2: Compute BERT embeddings
        # ============================================================
        print("[2/6] Computing BERT embeddings...")
        resume_embedding = self._get_embedding(processed_text)
        
        jd_cleaned = clean_text(job_description)
        jd_truncated = truncate_for_bert(jd_cleaned)
        jd_embedding = self._get_embedding(jd_truncated)
        
        # ============================================================
        # Step 3: Compute semantic similarity (Cosine Similarity)
        # ============================================================
        print("[3/6] Computing match score...")
        # Since both embeddings are normalized, dot product = cosine similarity
        similarity = torch.dot(resume_embedding, jd_embedding).item()
        match_score = round(max(0, similarity) * 100, 2)
        
        # ============================================================
        # Step 4: Skill extraction and gap analysis
        # ============================================================
        print("[4/6] Extracting skills...")
        resume_skills = extract_skills(raw_text)
        jd_skills = extract_skills_from_jd(job_description)
        
        matching_skills = sorted(resume_skills.intersection(jd_skills))
        missing_skills = sorted(jd_skills - resume_skills)
        
        skill_match_pct = get_skill_match_percentage(resume_skills, jd_skills)
        
        # ============================================================
        # Step 5: Classify role (optional enrichment)
        # ============================================================
        print("[5/6] Classifying role...")
        classification = self._classify_role(processed_text)
        
        # ============================================================
        # Step 6: Generate strengths and suggestions
        # ============================================================
        print("[6/6] Generating analysis report...")
        strengths = identify_resume_strengths(
            resume_skills,
            set(matching_skills),
            classification["predicted_role"]
        )
        suggestions = generate_improvement_suggestions(
            set(missing_skills),
            resume_skills
        )
        
        # Build result
        result = {
            "job_role": job_role,
            "match_score": match_score,
            "skill_match_percentage": round(skill_match_pct, 2),
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "resume_skills": sorted(resume_skills),
            "jd_skills": sorted(jd_skills),
            "resume_strengths": strengths,
            "improvement_suggestions": suggestions,
            "predicted_role": classification["predicted_role"],
            "predicted_confidence": classification["confidence"],
        }
        
        # Print formatted results
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: Dict):
        """Print a formatted analysis result."""
        print(f"\n{'=' * 60}")
        print("ANALYSIS RESULTS")
        print(f"{'=' * 60}")
        
        print(f"\n  Target Role:    {result['job_role']}")
        print(f"  Predicted Role: {result['predicted_role']} "
              f"({result['predicted_confidence']}%)")
        
        # Match Score
        bar = '#' * int(result['match_score'] / 2)
        print(f"\n  MATCH SCORE: {result['match_score']}%")
        print(f"  |{bar}|")
        
        print(f"  Skill Match: {result['skill_match_percentage']}%")
        
        # Matching Skills
        print(f"\n  MATCHING SKILLS ({len(result['matching_skills'])}):")
        for skill in result['matching_skills']:
            print(f"    [+] {skill}")
        
        # Missing Skills
        print(f"\n  MISSING SKILLS ({len(result['missing_skills'])}):")
        for skill in result['missing_skills']:
            print(f"    [-] {skill}")
        
        # Strengths
        print(f"\n  RESUME STRENGTHS:")
        for s in result['resume_strengths']:
            print(f"    * {s}")
        
        # Suggestions
        print(f"\n  IMPROVEMENT SUGGESTIONS:")
        for s in result['improvement_suggestions']:
            print(f"    > {s}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI RESUME ANALYZER - INFERENCE PIPELINE")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # ---- Test 1: Resume vs JD Analysis (NEW) ----
    print("\n" + "=" * 60)
    print("TEST: Resume vs Job Description Analysis")
    print("=" * 60)
    
    # Load a sample resume and JD
    test_resume = project_root / "data" / "resumes" / "resume_data_scientist_001.txt"
    test_jd_file = project_root / "data" / "job_descriptions" / "jd_data_scientist.txt"
    
    if test_resume.exists() and test_jd_file.exists():
        jd_text = load_text_file(str(test_jd_file))
        
        analyzer = ResumeJobAnalyzer()
        result = analyzer.analyze_against_job(
            resume_path=str(test_resume),
            job_description=jd_text,
            job_role="Data Scientist"
        )
        
        print("\n" + "=" * 60)
        print("JSON OUTPUT:")
        print("=" * 60)
        # Print the clean JSON output
        import json as json_module
        clean_output = {
            "job_role": result["job_role"],
            "match_score": result["match_score"],
            "matching_skills": result["matching_skills"],
            "missing_skills": result["missing_skills"],
            "resume_strengths": result["resume_strengths"],
            "improvement_suggestions": result["improvement_suggestions"]
        }
        print(json_module.dumps(clean_output, indent=2))
    else:
        print(f"Test files not found:")
        print(f"  Resume: {test_resume} (exists={test_resume.exists()})")
        print(f"  JD: {test_jd_file} (exists={test_jd_file.exists()})")
