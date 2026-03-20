# 🚀 AI Resume Analyzer

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Django-4.2+-green.svg" alt="Django">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg" alt="Docker">
</div>

<br>

A production-grade, deep learning-powered system that analyzes resumes against job descriptions using **BERT**. It goes beyond simple keyword matching by computing deep semantic similarity, identifying matching/missing skills, predicting candidate job roles, and generating actionable improvement suggestions. Complete with a **premium dark-mode web interface**, robust security, and Dockerized deployment.

---

## ✨ Features

### Deep Learning AI Core
| Feature | Technique used |
|---------|----------------|
| **Deep Semantic Match** | BERT `[CLS]` embeddings + Cosine similarity against Job Description |
| **Role Classification** | BERT fine-tuning + Linear classifier (Predicts out of 5 tech roles) |
| **Skill Gap Analysis** | Iterative keyword extraction + Set operations |
| **Smart Insights** | Heuristic domain mapping for strengths & actionable suggestions |
| **PDF Extraction** | Secure magic-byte verified PDF parsing via `pdfplumber` |

### Premium Web Interface
- **Dark-Mode Aesthetic**: Beautiful, modern UI with glowing gradients and fluid transitions.
- **Interactive Uploads**: Drag-and-drop resume upload zone with loading states.
- **Animated Results**: SVG score rings, animated coverage bars, and colored skill chips.
- **Fully Responsive**: Works seamlessly across desktop and mobile devices.

### Production & Security Ready
- **Security Hardened**: CSRF protection, secure cookies, XSS/Clickjacking headers, and HTTPS enforcement.
- **File Validation**: 5MB size limit, content-type validation, and PDF magic-byte checking.
- **Deployment Ready**: Included `Dockerfile` for Hugging Face Spaces and `render.yaml` for Render.
- **Optimized Serving**: WhiteNoise for static files and Gunicorn WSGI server.

---

## 🧠 System Architecture

```text
INPUT                              PROCESS                           OUTPUT
─────                              ───────                           ──────

Resume (PDF/TXT)                                                     
    │                                                                
    ├─→ Text Extraction                                              
    │   └─→ Cleaning ────→ BERT Tokenizer                              
    │                           │                                       
    │               ┌───────────┼──────────┐                             
    │               │           │          │                             
    │               v           v          v                             
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
    │   └─→ JD Embedding ──────────┤          │                      
    │       (768-dim)              │          │                      
    │                              │          │                      
    ├─→ Skill Extraction           │          │                      
    │   └─→ JD Skills ─────────────│──────────┤                      
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

---

## 🏗️ Project Structure

```text
resume_analyzer/
├── app/django_app/           # Django Web Application
│   ├── analyzer/             # Core app (views, forms, UI templates)
│   ├── resume_project/       # Project settings (security, WSGI)
│   └── manage.py
├── data/                     # Generated training data & labels
├── models/
│   └── bert_classifier.pt    # PyTorch Model Checkpoint (Use Git LFS)
├── scripts/                  # Data generation & Colab training scripts
├── src/                      # ML Source Code
│   ├── dataset.py            # PyTorch Dataset + BERT tokenization
│   ├── evaluation.py         # Metrics computing
│   ├── model.py              # BERT + classification head architecture
│   ├── predict.py            # Core Pipeline (ResumeJobAnalyzer)
│   ├── train.py              # Training loop (AdamW, CrossEntropyLoss)
│   └── utils.py              # Text preprocessing & skill extraction
├── Dockerfile                # Hugging Face Spaces Docker config
├── render.yaml               # Render Deployment config
├── build.sh                  # Render Build script
├── requirements.txt
└── .env.example              # Environment variables template
```

---

## ⚙️ Installation & Usage (Local Development)

### 1. Prerequisites
- Python 3.10+
- Git Large File Storage (Git LFS) installed

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/Ravinthra/resume_analyzer.git
cd resume_analyzer

# Setup Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Copy the example file and generate a Django Secret Key:
```bash
cp .env.example .env
```
Edit `.env` to include your secrets, setting `DJANGO_DEBUG=True` for local development.

### 4. Run the Web Interface

```bash
cd app/django_app
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.

---

## 🚀 Deployment

The project is configured for easy deployment on **Hugging Face Spaces** (Docker) and **Render**.

### Hugging Face Spaces (Recommended)
1. Create a physical space on HF (SDK: **Docker**).
2. Ensure you have `git-lfs` initialized.
3. Commit the 438MB `models/bert_classifier.pt` using Git LFS.
4. Push to Hugging Face:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/resume-analyzer
git push hf main
```
The provided `Dockerfile` installs CPU-only PyTorch (saving space), pre-downloads the BERT base model to improve cold-start times, and serves the app using Gunicorn on port `7860`.

### Render
Use the included `render.yaml` blueprint. Connect your GitHub repository to Render, and it will automatically provision the service using the `build.sh` script and `Procfile`.

---

## 🔬 Model Details

| Parameter | Value |
| --------- | ----- |
| **Base Model** | `bert-base-uncased` (110M params) |
| **Classification Head**| `Linear(768 → 5)` with `Dropout(0.3)` |
| **Optimizer** | AdamW (lr=2e-5) |
| **Loss Function** | CrossEntropyLoss |
| **Scheduler** | Linear warmup (10%) + decay |
| **Training Data** | 600+ synthetic resumes |
| **Validation F1** | 1.000 |

### Supported Job Roles (Classification)
0. Data Scientist
1. Software Engineer
2. Web Developer
3. DevOps Engineer
4. ML Engineer

---

## 📄 License

This project is for educational and portfolio purposes.

> *Built as a comprehensive deep learning portfolio project demonstrating BERT fine-tuning, transfer learning, robust ML pipeline design, and premium web application engineering.*
