# AI Resume Analyzer

A production-quality deep learning system that analyzes resumes against job descriptions using **BERT**. Computes semantic similarity, identifies matching and missing skills, and generates actionable improvement suggestions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Django](https://img.shields.io/badge/Django-4.2+-green.svg)

---

## Features

| Feature | Description | Technique |
| --------- | ------------- | ----------- |
| **Resume vs JD Analysis** | Compare a resume against a specific job description | BERT [CLS] embeddings + Cosine similarity |
| **Role Classification** | Predicts candidate's job role from resume text | BERT fine-tuning + Linear classifier |
| **Skill Gap Analysis** | Identifies matching and missing skills | Keyword extraction + Set operations |
| **Strengths & Suggestions** | Generates resume strengths and improvement tips | Skill domain mapping + categorization |
| **PDF Support** | Extracts text from PDF resumes | pdfplumber |
| **Web Interface** | Upload resume + paste JD via browser | Django |

## Architecture

```text
INPUT                              PROCESS                           OUTPUT
─────                              ───────                           ──────

Resume (PDF/TXT)                                                     
    │                                                                
    ├─→ Text Extraction                                              
    │   └─→ Cleaning ──→ BERT Tokenizer                              
    │                        │                                       
    │               ┌───────┼──────────┐                             
    │               │       │          │                             
    │               v       v          v                             
    │         Classification  Resume     Skill                       
    │         Head            Embedding   Extraction                  
    │         (Linear)        (768-dim)   (Regex)                     
    │               │              │          │                      
    │               v              │          v                      
    │         Predicted Role       │     Resume Skills               
    │                              │          │                      
Job Description                    │          │                      
    │                              │          │                      
    ├─→ BERT Encoding              │          │                      
    │   └─→ JD Embedding ─────────┤          │                      
    │       (768-dim)              │          │                      
    │                              │          │                      
    ├─→ Skill Extraction           │          │                      
    │   └─→ JD Skills ────────────│──────────┤                      
    │                              │          │                      
    │                              v          v                      
    │                        Cosine      Set Operations              
    │                        Similarity  (∩ and -)                   
    │                              │          │                      
    v                              v          v                      
                                                                     
                             Match Score    Matching Skills           
                             (0-100%)       Missing Skills            
                                           Strengths                 
                                           Suggestions               
```

## Deep Learning Concepts

This project demonstrates the following concepts:

- **Transfer Learning** — Fine-tuning pretrained BERT on a downstream task
- **BERT [CLS] Embeddings** — 768-dim semantic representations of documents
- **Neural Network Classification Head** — Linear layer mapping embeddings to class logits
- **Cosine Similarity** — Measuring semantic alignment between resume and JD embeddings
- **CrossEntropyLoss** — Standard multi-class classification loss
- **AdamW Optimizer** — Weight-decoupled Adam for Transformer fine-tuning
- **Learning Rate Scheduling** — Linear warmup + decay
- **Dropout Regularization** — Preventing overfitting
- **Backpropagation** — Gradient computation via chain rule

## Project Structure

```text
resume_analyzer/
├── data/
│   ├── resumes/              # 600+ synthetic resumes (120 per role)
│   ├── job_descriptions/     # Job description files (5 roles)
│   ├── dataset.csv           # Resume-to-label mapping
│   └── labels.json           # Label schema (5 classes)
├── models/
│   └── bert_classifier.pt    # Trained model checkpoint
├── scripts/
│   └── generate_resumes.py   # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── utils.py              # Text preprocessing, skill extraction, suggestions
│   ├── dataset.py            # PyTorch Dataset + BERT tokenization
│   ├── model.py              # BERT + classification head architecture
│   ├── train.py              # Training loop (AdamW, CrossEntropyLoss)
│   └── predict.py            # ResumeJobAnalyzer + ResumeAnalyzer pipelines
├── app/
│   └── django_app/           # Django web interface
│       ├── analyzer/         # Resume analyzer app (forms, views, templates)
│       └── resume_project/   # Django project settings
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Ravinthra/resume_analyzer.git
cd resume_analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
cd resume_analyzer
python -m src.train
```

This will:

- Load 600+ resumes from `data/resumes/`
- Tokenize with BERT's WordPiece tokenizer
- Train for 3 epochs with AdamW optimizer
- Save the best model to `models/bert_classifier.pt`

### 2. Analyze Resume vs Job Description (CLI)

```bash
python -m src.predict
```

**Output format:**

```json
{
  "job_role": "Data Scientist",
  "match_score": 82,
  "matching_skills": ["python", "pytorch", "pandas"],
  "missing_skills": ["spark", "kubernetes"],
  "resume_strengths": ["Strong ML & AI expertise"],
  "improvement_suggestions": ["Add Big Data Technologies experience"]
}
```

### 3. Run the Web Interface

```bash
cd app/django_app
python manage.py runserver
```

Open `http://localhost:8000/` → Upload resume + Enter job role + Paste JD → Get results.

## Model Details

| Parameter | Value |
| ----------- | ------- |
| Base Model | `bert-base-uncased` (110M params) |
| Classification Head | Linear(768 → 5) |
| Dropout | 0.3 |
| Optimizer | AdamW (lr=2e-5) |
| Loss Function | CrossEntropyLoss |
| Scheduler | Linear warmup (10%) + decay |
| Max Sequence Length | 256 tokens |
| Batch Size | 8 |
| Training Data | 600+ synthetic resumes |
| Val F1 Score | 1.000 |

## Job Role Classes

| ID | Label |
| ---- | ------- |
| 0 | Data Scientist |
| 1 | Software Engineer |
| 2 | Web Developer |
| 3 | DevOps Engineer |
| 4 | ML Engineer |

## Key Files Explained

### `src/predict.py`

- `ResumeJobAnalyzer` — Analyzes resume against a specific JD (match score, skill gaps, suggestions)
- `ResumeAnalyzer` — Classifies resume role and matches against stored JDs

### `src/utils.py`

Text preprocessing (PDF extraction, regex cleaning) and skill analysis (extraction, gap detection, strengths identification, improvement suggestions).

### `src/model.py`

`ResumeClassifier(nn.Module)` — BERT encoder + Dropout + Linear head with `freeze_bert()` / `unfreeze_bert()` for transfer learning.

### `src/train.py`

Training loop with AdamW optimizer, CrossEntropyLoss, learning rate warmup, gradient clipping, and checkpointing.

## Technologies

- **PyTorch** — Deep learning framework
- **HuggingFace Transformers** — Pretrained BERT model and tokenizer
- **pdfplumber** — PDF text extraction
- **scikit-learn** — Evaluation metrics
- **Django** — Web interface
- **pandas/numpy** — Data handling

## License

This project is for educational and portfolio purposes.

---

*Built as a deep learning portfolio project demonstrating BERT fine-tuning, transfer learning, and production ML pipeline design.*
