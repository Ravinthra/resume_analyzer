"""
utils.py — Data Preprocessing Utilities for AI Resume Analyzer
================================================================

This module handles three critical preprocessing tasks:
1. PDF Text Extraction — Converts resume PDFs to raw text
2. Text Cleaning       — Normalizes text for BERT tokenization
3. Skill Extraction    — Identifies skills from resume/JD text

Why preprocessing matters for BERT:
- BERT's tokenizer handles most text normalization (lowercasing, subword splitting)
- But we still need to remove noise (URLs, emails, special formatting) that
  wastes BERT's 512-token input limit
- Cleaner input → better [CLS] embeddings → more accurate classification

Author: AI Resume Analyzer Project
"""

import re
import os
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional

# Configure logging for debugging preprocessing issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: PDF TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pdfplumber.
    
    Why pdfplumber over PyPDF2?
    - Better handling of complex PDF layouts (tables, columns)
    - More accurate text extraction with position-aware parsing
    - Handles scanned PDFs better when combined with OCR
    
    Args:
        pdf_path: Absolute or relative path to the PDF file
        
    Returns:
        Extracted text as a single string
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF parsing fails
    """
    import pdfplumber
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    text_parts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.warning(f"No text extracted from page {page_num + 1}")
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise
    
    full_text = "\n".join(text_parts)
    logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
    
    return full_text


def load_text_file(file_path: str) -> str:
    """
    Load text from a plain text file (.txt).
    
    Used for loading sample resumes and job descriptions
    during development when working with .txt files.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        File contents as a string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# SECTION 2: TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize resume/JD text for BERT processing.
    
    Cleaning strategy:
    - BERT handles lowercasing internally (bert-base-uncased)
    - We focus on removing noise that wastes the 512-token budget:
      • URLs, emails, phone numbers (not semantically useful for classification)
      • Excessive whitespace and special characters
      • Non-ASCII characters that BERT maps to [UNK]
    
    What we KEEP:
    - Technical terms, skill names, job titles
    - Sentence structure (BERT uses positional context)
    - Numbers (years of experience matter)
    
    Args:
        text: Raw text extracted from resume or JD
        
    Returns:
        Cleaned text string ready for BERT tokenizer
    """
    if not text or not text.strip():
        return ""
    
    # Remove URLs (http/https/www links)
    # These waste tokens and don't help classification
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # Remove email addresses
    # Not useful for determining job role
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "", text)
    
    # Remove special characters but keep essential punctuation
    # Keep: letters, digits, spaces, periods, commas, hyphens, slashes, plus signs
    # These are important for terms like "C++", "Node.js", "CI/CD"
    text = re.sub(r"[^a-zA-Z0-9\s.,\-/+#]", " ", text)
    
    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_for_bert(text: str, max_words: int = 400) -> str:
    """
    Truncate text to fit within BERT's 512-token limit.
    
    Why 400 words instead of 512?
    - BERT tokenizer uses WordPiece → 1 word often becomes 2-3 tokens
    - Example: "microservices" → ["micro", "##services"] (2 tokens)
    - We also need room for [CLS] and [SEP] special tokens
    - 400 words ≈ 450-500 tokens — safe margin
    
    Args:
        text: Cleaned text string
        max_words: Maximum number of words to keep (default 400)
        
    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) > max_words:
        logger.info(f"Truncating text from {len(words)} to {max_words} words")
        words = words[:max_words]
    return " ".join(words)


# ============================================================================
# SECTION 3: SKILL EXTRACTION
# ============================================================================

# Comprehensive skill dictionary organized by category
# This is used for both skill extraction and gap analysis
SKILL_DATABASE = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "go", "golang", "c++", "cpp",
    "c#", "csharp", "rust", "ruby", "scala", "r", "julia", "swift", "kotlin",
    "php", "perl", "bash", "shell", "sql", "matlab",
    
    # ML/DL Frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "xgboost",
    "lightgbm", "jax", "onnx", "caffe", "mxnet",
    
    # NLP
    "bert", "gpt", "transformers", "hugging face", "spacy", "nltk",
    "word2vec", "glove", "t5", "langchain", "llm",
    
    # Deep Learning Concepts
    "cnn", "rnn", "lstm", "gan", "autoencoder", "transformer",
    "attention mechanism", "transfer learning", "reinforcement learning",
    "computer vision", "natural language processing", "nlp",
    
    # Data Science
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "tableau", "power bi", "statistics", "a/b testing",
    "hypothesis testing", "regression", "classification", "clustering",
    "feature engineering", "data visualization",
    
    # Big Data
    "spark", "hadoop", "kafka", "flink", "airflow", "dbt",
    "hive", "presto", "beam",
    
    # Cloud
    "aws", "gcp", "azure", "sagemaker", "ec2", "s3", "lambda",
    "google cloud", "cloud computing",
    
    # DevOps / Infrastructure
    "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "github actions", "gitlab ci", "ci/cd", "helm", "istio",
    "prometheus", "grafana", "datadog", "elk stack",
    
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "firebase", "sqlite", "neo4j",
    
    # Web Development
    "react", "react.js", "reactjs", "vue", "vue.js", "vuejs",
    "angular", "next.js", "nextjs", "nuxt.js", "node.js", "nodejs",
    "express", "django", "flask", "fastapi", "spring boot",
    "html", "html5", "css", "css3", "tailwind", "sass",
    "webpack", "vite", "graphql", "rest", "restful",
    
    # MLOps
    "mlflow", "kubeflow", "weights and biases", "wandb", "dvc",
    "model serving", "torchserve", "triton", "bentoml",
    
    # General
    "git", "linux", "agile", "scrum", "microservices",
    "distributed systems", "system design", "data structures",
    "algorithms", "machine learning", "deep learning",
}


def extract_skills(text: str) -> Set[str]:
    """
    Extract technical skills mentioned in a resume or job description.
    
    Approach:
    - Convert text to lowercase for case-insensitive matching
    - Check for each skill in our curated skill database
    - Use word boundary matching to avoid partial matches
      (e.g., "class" shouldn't match inside "classification")
    
    Why keyword matching over NER?
    - NER models need labeled training data for technical skills
    - Keyword matching is deterministic and explainable
    - Easy to extend by adding new skills to the database
    - Fast — no GPU needed for skill extraction
    
    Args:
        text: Resume or JD text (cleaned or raw)
        
    Returns:
        Set of identified skills (lowercase)
    """
    text_lower = text.lower()
    found_skills = set()
    
    for skill in SKILL_DATABASE:
        # Use word boundary regex for accurate matching
        # \b ensures we match whole words/phrases only
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    return found_skills


def find_missing_skills(resume_skills: Set[str], jd_skills: Set[str]) -> Set[str]:
    """
    Identify skills required by a job description but missing from a resume.
    
    This is a simple but powerful technique:
    - Set difference: JD_skills - Resume_skills = Missing skills
    - Tells the candidate exactly what to learn/add
    
    Args:
        resume_skills: Skills found in the resume
        jd_skills: Skills required by the job description
        
    Returns:
        Set of skills in JD but not in resume
    """
    return jd_skills - resume_skills


def get_skill_match_percentage(resume_skills: Set[str], jd_skills: Set[str]) -> float:
    """
    Calculate what percentage of required JD skills the resume covers.
    
    Formula: (matched_skills / total_jd_skills) × 100
    
    Args:
        resume_skills: Skills found in the resume
        jd_skills: Skills required by the job description
        
    Returns:
        Percentage (0.0 to 100.0) of JD skills covered
    """
    if not jd_skills:
        return 0.0
    
    matched = resume_skills.intersection(jd_skills)
    return (len(matched) / len(jd_skills)) * 100.0


def extract_skills_from_jd(jd_text: str) -> Set[str]:
    """
    Extract required skills from a job description.
    
    This is a convenience wrapper around extract_skills() that first
    cleans and normalizes the JD text before extraction.
    
    Args:
        jd_text: Raw job description text (pasted by user)
        
    Returns:
        Set of skills found in the job description
    """
    cleaned = clean_text(jd_text)
    return extract_skills(cleaned)


def identify_resume_strengths(resume_skills: Set[str], 
                               matching_skills: Set[str],
                               predicted_role: str = None) -> List[str]:
    """
    Identify resume strengths based on skills and predicted role.
    
    Maps matching skills to high-level strength categories.
    This provides interpretable feedback beyond just listing skills.
    
    Args:
        resume_skills: All skills found in the resume
        matching_skills: Skills that match the target JD
        predicted_role: Optional predicted role from classifier
        
    Returns:
        List of strength descriptions
    """
    strengths = []
    
    # Skill domain mapping — groups of related skills
    skill_domains = {
        "Machine Learning & AI": {
            "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
            "xgboost", "lightgbm", "jax", "machine learning", "deep learning",
            "neural networks", "cnn", "rnn", "lstm", "transformer"
        },
        "NLP & Language Models": {
            "bert", "gpt", "transformers", "hugging face", "spacy", "nltk",
            "nlp", "natural language processing", "langchain", "llm", "t5"
        },
        "Cloud Computing": {
            "aws", "gcp", "azure", "sagemaker", "ec2", "s3", "lambda",
            "google cloud", "cloud computing"
        },
        "DevOps & Infrastructure": {
            "docker", "kubernetes", "terraform", "ansible", "jenkins",
            "ci/cd", "github actions", "gitlab ci", "helm", "istio"
        },
        "Data Engineering": {
            "spark", "hadoop", "kafka", "airflow", "hive", "presto",
            "flink", "beam", "dbt"
        },
        "Web Development": {
            "react", "vue", "angular", "next.js", "node.js", "django",
            "flask", "fastapi", "html", "css", "javascript", "typescript"
        },
        "Data Analysis & Visualization": {
            "pandas", "numpy", "matplotlib", "seaborn", "plotly",
            "tableau", "power bi", "data visualization", "statistics"
        },
        "Database Management": {
            "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "cassandra", "dynamodb", "firebase", "sqlite"
        },
        "Programming Proficiency": {
            "python", "java", "c++", "go", "rust", "scala", "r"
        },
    }
    
    # Check which domains the resume covers
    for domain_name, domain_skills in skill_domains.items():
        overlap = resume_skills.intersection(domain_skills)
        if len(overlap) >= 3:
            strengths.append(f"Strong {domain_name} expertise ({', '.join(sorted(overlap)[:4])})")
        elif len(overlap) >= 1:
            strengths.append(f"{domain_name} experience ({', '.join(sorted(overlap))})")
    
    # Add match-rate based strength
    if matching_skills:
        match_rate = len(matching_skills)
        if match_rate >= 8:
            strengths.append(f"Excellent JD alignment — {match_rate} matching skills")
        elif match_rate >= 5:
            strengths.append(f"Good JD alignment — {match_rate} matching skills")
    
    # Add role-specific strengths
    if predicted_role:
        strengths.append(f"Profile aligns with {predicted_role} role")
    
    return strengths if strengths else ["Broad technical background"]


def generate_improvement_suggestions(missing_skills: Set[str],
                                      resume_skills: Set[str]) -> List[str]:
    """
    Generate actionable improvement suggestions based on skill gaps.
    
    Groups missing skills by category and produces specific,
    helpful recommendations the candidate can act on.
    
    Args:
        missing_skills: Skills required by JD but missing from resume
        resume_skills: Skills found in the resume
        
    Returns:
        List of improvement suggestion strings
    """
    suggestions = []
    
    if not missing_skills:
        suggestions.append("Your resume covers all required skills — great match!")
        return suggestions
    
    # Group missing skills by domain for actionable suggestions
    skill_categories = {
        "Programming Languages": {
            "python", "java", "javascript", "typescript", "go", "golang",
            "c++", "cpp", "c#", "rust", "ruby", "scala", "r", "kotlin", "swift"
        },
        "ML/AI Frameworks": {
            "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
            "xgboost", "lightgbm", "jax", "onnx"
        },
        "Cloud Platforms": {
            "aws", "gcp", "azure", "sagemaker", "google cloud", "cloud computing"
        },
        "DevOps Tools": {
            "docker", "kubernetes", "terraform", "ansible", "jenkins",
            "ci/cd", "github actions", "gitlab ci", "helm"
        },
        "Big Data Technologies": {
            "spark", "hadoop", "kafka", "airflow", "hive", "flink"
        },
        "Database Systems": {
            "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "cassandra", "dynamodb"
        },
        "Web Technologies": {
            "react", "vue", "angular", "next.js", "node.js", "django",
            "flask", "fastapi", "graphql", "rest"
        },
    }
    
    for category, category_skills in skill_categories.items():
        missing_in_category = missing_skills.intersection(category_skills)
        if missing_in_category:
            skill_list = ", ".join(sorted(missing_in_category))
            suggestions.append(f"Add {category} experience: {skill_list}")
    
    # Generic missing skills not in any category
    categorized = set()
    for cat_skills in skill_categories.values():
        categorized.update(cat_skills)
    uncategorized_missing = missing_skills - categorized
    if uncategorized_missing:
        skill_list = ", ".join(sorted(uncategorized_missing)[:5])
        suggestions.append(f"Consider adding: {skill_list}")
    
    # Overall guidance
    if len(missing_skills) > 5:
        suggestions.append(
            f"Focus on the top {min(5, len(missing_skills))} most critical skills "
            f"for this role to maximize your match score"
        )
    
    return suggestions


# ============================================================================
# SECTION 4: LABEL UTILITIES
# ============================================================================

def load_labels(labels_path: str) -> Dict:
    """
    Load the label schema from labels.json.
    
    Args:
        labels_path: Path to labels.json
        
    Returns:
        Dictionary containing label_to_id, id_to_label, and num_classes
    """
    with open(labels_path, "r") as f:
        return json.load(f)


# ============================================================================
# SECTION 5: FULL PREPROCESSING PIPELINE
# ============================================================================

def preprocess_resume(file_path: str) -> str:
    """
    Complete preprocessing pipeline: Load → Clean → Truncate.
    
    This is the main entry point for processing any resume file.
    Handles both PDF and text files automatically.
    
    Args:
        file_path: Path to resume file (.pdf or .txt)
        
    Returns:
        Cleaned, truncated text ready for BERT tokenizer
    """
    # Step 1: Extract raw text based on file type
    if file_path.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    else:
        raw_text = load_text_file(file_path)
    
    # Step 2: Clean the text (remove noise)
    cleaned = clean_text(raw_text)
    
    # Step 3: Truncate to fit BERT's context window
    truncated = truncate_for_bert(cleaned)
    
    logger.info(f"Preprocessed resume: {len(raw_text)} chars → {len(truncated)} chars")
    
    return truncated


# ============================================================================
# Quick test — run this file directly to verify preprocessing works
# ============================================================================

if __name__ == "__main__":
    # Test with a sample resume
    project_root = Path(__file__).parent.parent
    sample_resume = project_root / "data" / "resumes" / "resume_data_scientist_01.txt"
    
    if sample_resume.exists():
        print("=" * 60)
        print("PREPROCESSING PIPELINE TEST")
        print("=" * 60)
        
        # Test the full pipeline
        processed = preprocess_resume(str(sample_resume))
        print(f"\n📄 Input file: {sample_resume.name}")
        print(f"📏 Processed length: {len(processed)} characters")
        print(f"\n📝 First 300 chars:\n{processed[:300]}...")
        
        # Test skill extraction
        raw_text = load_text_file(str(sample_resume))
        skills = extract_skills(raw_text)
        print(f"\n🔧 Skills found ({len(skills)}):")
        for skill in sorted(skills):
            print(f"   • {skill}")
        
        # Test with a job description
        sample_jd = project_root / "data" / "job_descriptions" / "jd_data_scientist.txt"
        if sample_jd.exists():
            jd_text = load_text_file(str(sample_jd))
            jd_skills = extract_skills(jd_text)
            
            missing = find_missing_skills(skills, jd_skills)
            match_pct = get_skill_match_percentage(skills, jd_skills)
            
            print(f"\n📊 Skill Match: {match_pct:.1f}%")
            print(f"\n❌ Missing skills ({len(missing)}):")
            for skill in sorted(missing):
                print(f"   • {skill}")
    else:
        print(f"Sample resume not found at {sample_resume}")
