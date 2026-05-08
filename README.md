# 🚀 AI Resume Analyzer

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Django](https://img.shields.io/badge/Django-4.2+-092E20?style=for-the-badge&logo=django&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-Fine--Tuned-FF6F00?style=for-the-badge)

A production-grade, deep learning–powered system that analyzes resumes against job descriptions using fine-tuned **BERT**. It goes far beyond keyword matching — computing deep semantic similarity via BERT embeddings, predicting a candidate's job role, performing skill gap analysis, and generating actionable career improvement suggestions.

The system ships with a **premium dark-mode web interface**, robust security hardening, and a **Dockerized deployment** targeting Hugging Face Spaces.

---

## ✨ Feature Overview

### 🤖 Deep Learning AI Core

| Feature | Technique |
| --- | --- |
| **Semantic Match Score** | BERT `[CLS]` embeddings + Cosine Similarity against the target Job Description |
| **Role Classification** | BERT fine-tuned with a `Linear(768 → 25)` classification head — predicts **25 IT professions** |
| **Skill Gap Analysis** | Regex-based keyword extraction from a curated 200+ skill database + set operations |
| **Strengths & Suggestions** | Heuristic domain mapping — groups skills into meaningful categories, generates targeted suggestions |
| **PDF/TXT Extraction** | Magic-byte verified PDF parsing via `pdfplumber`; plain text support for `.txt` resumes |
| **Candidate Ranking** | Batch BERT embedding + cosine ranking for screening multiple resumes against one JD |

### 🎨 Premium Web Interface

- **Dark-Mode UI**: Modern design with glowing gradients, animated score rings, and fluid transitions.
- **Drag-and-Drop Upload**: Interactive resume upload zone with live loading state feedback.
- **Animated Results Page**: Circular SVG match score, animated skill coverage bars, and color-coded skill chips.
- **Fully Responsive**: Seamless experience on both desktop and mobile browsers.

### 🛡️ Production & Security Ready

- **Security Hardened**: CSRF protection, HSTS, XSS/Clickjacking headers, `SameSite=None` cookie policy for Hugging Face Spaces iframes.
- **File Validation**: 5 MB size limit, content-type checking, and PDF magic-byte (`%PDF-`) verification.
- **Deployment Ready**: Included `Dockerfile` for Hugging Face Spaces (port `7860`, CPU-optimized PyTorch).
- **Optimized Serving**: WhiteNoise for static files, Gunicorn WSGI server with 2 workers.

---

## 🧠 System Architecture

### Full Inference Pipeline

```text
INPUT                           PROCESS                             OUTPUT
─────                           ───────                             ──────

Resume (PDF / TXT)
    │
    ├─→ 1. Text Extraction (pdfplumber / open())
    │        └─→ clean_text()                ←── removes URLs, emails,
    │              └─→ truncate_for_bert()        phone numbers, noise
    │                      │
    │                      │ (preprocessed text, ≤400 words)
    │                      ▼
    │              BERT Tokenizer (bert-base-uncased)
    │              input_ids + attention_mask
    │                      │
    │         ┌────────────┼─────────────────┐
    │         │            │                 │
    │         ▼            ▼                 ▼
    │   [Classifier]  [Embedding          [Skill
    │    BERT+Linear   Model]              Extraction]
    │    (fine-tuned)  bert-base-uncased   (regex +
    │         │        [CLS] → 768-dim     SKILL_DATABASE)
    │         │        vector, L2-norm     │
    │         ▼             │              ▼
    │   Softmax(logits)     │        Resume Skills (set)
    │   → Role Probs        │              │
    │   → Predicted Role    │              │
    │                       │              │
Job Description (text)      │              │
    │                       │              │
    ├─→ clean_text()        │              │
    │   truncate_for_bert() │              │
    │   BERT Tokenizer      │              │
    │   BertModel → [CLS]   │              │
    │   L2-normalize ───────┘              │
    │                       │              │
    ├─→ Skill Extraction ───│──────────────┤
    │   JD Skills (set)     │              │
    │                       ▼              ▼
    │              Cosine Similarity   Set Operations
    │              (dot product of     resume ∩ jd  → Matching
    │               unit vectors)      jd - resume  → Missing
    │                       │
    │               Match Score (0–100%)
    │
    └─────────────────────────────────────────────────────────────▶
                                                        {
                                                          job_role,
                                                          match_score,
                                                          matching_skills,
                                                          missing_skills,
                                                          resume_strengths,
                                                          improvement_suggestions,
                                                          classification: {
                                                            predicted_role,
                                                            confidence,
                                                            all_probabilities
                                                          }
                                                        }
```

### Two Prediction Classes

| Class | Purpose | Use Case |
| --- | --- | --- |
| `ResumeJobAnalyzer` | Analyze one resume against a **user-supplied** JD | Web interface, production |
| `ResumeAnalyzer` | Classify resume + match against **stored JDs** | Batch analysis, research |

---

## 🏗️ Project Structure

```text
resume_analyzer/
│
├── app/                          # Web Application
│   └── django_app/
│       ├── analyzer/             # Django app
│       │   ├── views.py          # Upload handler + ML pipeline call
│       │   ├── forms.py          # ResumeAnalysisForm (file + job role + JD)
│       │   ├── urls.py           # Route: / (home) and /analyze/ (POST)
│       │   ├── templates/
│       │   │   └── analyzer/
│       │   │       ├── home.html   # Dark-mode upload form
│       │   │       └── results.html # Animated results page
│       │   └── ...
│       └── resume_project/       # Django project config
│           ├── settings.py       # Security, WhiteNoise, CSRF, HSTS
│           ├── urls.py
│           ├── wsgi.py
│           └── asgi.py
│
├── src/                          # ML Source Code
│   ├── __init__.py
│   ├── model.py                  # ResumeClassifier — BERT + Linear(768→25)
│   ├── predict.py                # ResumeJobAnalyzer + ResumeAnalyzer pipelines
│   ├── train.py                  # Training loop (AdamW, CrossEntropy, LR schedule)
│   ├── dataset.py                # ResumeDataset + 3-way train/val/test split
│   ├── evaluation.py             # Accuracy, F1, confusion matrix, error analysis
│   ├── utils.py                  # Text cleaning, skill extraction, strength/suggestion generators
│   ├── ranking.py                # CandidateRanker — batch BERT cosine ranking
│   └── visualization.py         # Training curve plotting
│
├── scripts/                      # Data Generation & Training Scripts
│   ├── generate_resumes.py       # Generates 600 diverse synthetic resumes (5 classes)
│   ├── generate_90k_resumes.py   # Generates 90,000+ resumes across 25 IT classes
│   ├── role_configs.py           # Per-role skill pools, project templates, certifications
│   ├── role_data.py              # Extended role data and resume templates
│   └── colab_train.py            # Full Google Colab training script (FP16, T4 GPU)
│
├── notebooks/
│   └── colab_train_file.ipynb    # Jupyter notebook version of Colab training
│
├── data/
│   ├── labels.json               # 25-class label schema (label_to_id + id_to_label)
│   ├── dataset.csv               # Metadata: resume paths + label IDs
│   ├── job_descriptions/         # Pre-stored JD files (one per role class)
│   ├── synthetic_resumes/        # Generated synthetic resumes (organized by class)
│   ├── real_resumes/             # Real-world resume samples
│   └── metadata/                 # dataset.csv for 90K pipeline
│
├── models/
│   └── bert_classifier.pt        # PyTorch checkpoint (438 MB — tracked with Git LFS)
│
├── Dockerfile                    # HF Spaces Docker config (CPU PyTorch, port 7860)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore
└── pyrightconfig.json            # Python type-checking configuration
```

---

## ⚙️ Installation & Local Development

### Prerequisites

- Python **3.10+**
- **Git LFS** (for the 438 MB model checkpoint)
- `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/Ravinthra/resume_analyzer.git
cd resume_analyzer

# Pull the model checkpoint via Git LFS
git lfs install
git lfs pull
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run will download the `bert-base-uncased` model (~440 MB) from Hugging Face Hub.

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and set:

```env
# Generate with: python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
DJANGO_SECRET_KEY=your-secret-key-here

# Set True for local dev, False for production
DJANGO_DEBUG=True

DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
```

### 5. Run Database Migrations

```bash
cd app/django_app
python manage.py makemigrations
python manage.py migrate
```

### 6. Start the Development Server

```bash
python manage.py runserver
```

Open **<http://127.0.0.1:8000/>** in your browser.

---

## 🔬 ML Pipeline Deep Dive

### Model Architecture — `src/model.py`

```text
Input (input_ids [batch, 512], attention_mask [batch, 512])
    │
    ▼
BERT Encoder (bert-base-uncased)
    • 12 Transformer layers
    • 12 attention heads per layer
    • 768-dim hidden state
    • ~110M parameters
    │
    ▼
pooler_output — [CLS] token embedding [batch, 768]
    │
    ▼
Dropout(p=0.3)   ← regularization
    │
    ▼
Linear(768 → 25)  ← classification head
    │
    ▼
Logits [batch, 25]  →  Softmax  →  Predicted Role
```

### Semantic Similarity — `src/predict.py`

Cosine similarity is used to score how well a resume matches a job description:

```text
cos(θ) = (resume_embedding · jd_embedding) / (|resume_emb| × |jd_emb|)
```

Because both embeddings are L2-normalized to unit length, this simplifies to a plain **dot product** — efficient and numerically stable. The resulting score is shifted from `[-1, 1]` to `[0, 100]` for display.

### Skill Extraction — `src/utils.py`

A curated database of **200+ technical skills** (`SKILL_DATABASE`) is matched using **word-boundary regex** (`\b<skill>\b`) to avoid false positives (e.g., `class` not matching inside `classification`). Skills are grouped into domains including:

- Programming Languages (Python, Java, Go, Rust, …)
- ML/DL Frameworks (PyTorch, TensorFlow, JAX, …)
- NLP / LLMs (BERT, GPT, Transformers, LangChain, …)
- Cloud Platforms (AWS, GCP, Azure, SageMaker, …)
- DevOps / Infrastructure (Docker, Kubernetes, Terraform, …)
- Databases (PostgreSQL, MongoDB, Redis, …)
- Web Technologies (React, Next.js, Django, FastAPI, …)
- Data Engineering (Spark, Kafka, Airflow, dbt, …)
- MLOps (MLflow, Kubeflow, W&B, DVC, …)

---

## 📊 Model Details

| Parameter | Value |
| --- | --- |
| **Base Model** | `bert-base-uncased` (110M parameters) |
| **Classification Head** | `Linear(768 → 25)` with `Dropout(0.3)` |
| **Optimizer** | AdamW (lr=2e-5, weight_decay=0.01) |
| **Loss Function** | CrossEntropyLoss |
| **LR Scheduler** | Linear warmup (6%) + linear decay |
| **Batch Size** | 8 (local) / 48 (Colab T4 with FP16) |
| **Max Sequence Length** | 384 tokens |
| **Training Epochs** | 5 |
| **Training Data** | 90,000+ synthetic + real resumes |
| **Checkpoint Size** | 438 MB (`.pt` format) |

### Supported Job Roles — 25 Classes

| ID | Role | ID | Role |
| --- | --- | --- | --- |
| 0 | Data Scientist | 13 | Cloud Architect |
| 1 | Software Engineer | 14 | QA Engineer |
| 2 | Web Developer | 15 | Mobile App Developer |
| 3 | DevOps Engineer | 16 | Business Analyst |
| 4 | ML Engineer | 17 | IT Project Manager |
| 5 | Data Analyst | 18 | UI/UX Designer |
| 6 | Backend Developer | 19 | Data Engineer |
| 7 | Frontend Developer | 20 | Blockchain Developer |
| 8 | Full Stack Developer | 21 | Embedded Systems Engineer |
| 9 | Database Administrator | 22 | Game Developer |
| 10 | System Administrator | 23 | IT Support Specialist |
| 11 | Network Engineer | 24 | AI Research Scientist |
| 12 | Cybersecurity Analyst | | |

---

## 🗃️ Dataset & Training Data

### Data Generation Pipeline

The project includes two fully-featured synthetic resume generators:

**`scripts/generate_resumes.py`** — Generates 600 resumes (120 per class) across 5 classes:

- Diverse candidate profiles: 80+ first names, 70+ last names, 33+ top universities, 20+ degree types
- 50+ tech companies per role pool (FAANG, startups, consulting)
- 4 different resume format templates (traditional, modern, skills-first, narrative)
- 8–18 role-specific skills per resume
- 2–4 realistic project descriptions per resume
- 1–3 certifications per resume


**`scripts/generate_90k_resumes.py`** — Scales to 90,000+ resumes across all 25 IT job roles, using extended role configs from `scripts/role_configs.py`.

### Dataset Split Strategy

```text
Synthetic resumes (15,000)  →  80% train  |  20% val  |   0% test
Real resumes      (75,000)  →  70% train  |  15% val  |  15% test
                                                        ▲
                                             (real-only test set
                                              for honest evaluation)
```

### Training in Google Colab

The `scripts/colab_train.py` script provides a complete Colab training pipeline optimized for the **T4 GPU (15 GB VRAM)**:

- **FP16 Mixed Precision** — halves memory usage, nearly doubles throughput
- **Gradient Accumulation** (steps=2) — effective batch size of 96
- **Full BERT Fine-Tuning** — all 110M parameters are trainable
- **Automatic dataset generation** if metadata CSV is missing
- Expected runtime: ~20 minutes on a free T4 Colab instance

```bash
# In Google Colab (after cloning the repo):
!python scripts/colab_train.py
```

The `notebooks/colab_train_file.ipynb` notebook provides the same workflow in an interactive format.

---

## 🌐 Web Application

### How It Works

1. User uploads a **PDF or TXT** resume and pastes a **Job Description** with a selected **Job Role**
2. Django view (`analyzer/views.py`) validates the file (type, size, magic bytes) and saves to a temp file
3. The `ResumeJobAnalyzer` is lazy-loaded on first request and **cached globally** (avoids reloading BERT on every request)
4. The pipeline runs:
   - Text extraction + preprocessing
   - BERT embedding computation for resume and JD
   - Cosine similarity → match score
   - Skill extraction via regex on both documents
   - Role classification with confidence scores
   - Strengths and improvement suggestions generation
5. Results are rendered in `results.html` with animated visualizations

### Security Measures

| Measure | Implementation |
| --- | --- |
| **CSRF Protection** | Django `CsrfViewMiddleware` + `CSRF_COOKIE_SAMESITE='None'` for HF iframe |
| **File Type Validation** | Extension check (`.pdf` / `.txt`) + PDF magic bytes (`%PDF-`) |
| **File Size Limit** | 5 MB max (`FILE_UPLOAD_MAX_MEMORY_SIZE`) |
| **XSS Prevention** | `SECURE_BROWSER_XSS_FILTER`, `SECURE_CONTENT_TYPE_NOSNIFF` |
| **HSTS** | `SECURE_HSTS_SECONDS=31536000` (production only) |
| **Secure Cookies** | `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`, `CSRF_COOKIE_HTTPONLY` |
| **Temp File Cleanup** | `finally` block ensures temp files are always deleted |
| **Non-Root Docker** | Runs as `appuser` (UID 1000) per Hugging Face Spaces policy |

### Django Form — 25 Job Role Choices

The `ResumeAnalysisForm` (`analyzer/forms.py`) accepts:

- **resume_file** — `FileField` (PDF or TXT, max 5 MB)
- **job_role** — `ChoiceField` with all 25 IT role options
- **job_description** — `CharField` (textarea, paste full JD text)


---

## 🚀 Deployment

### Hugging Face Spaces (Docker) — Recommended

The included `Dockerfile` is pre-configured for Hugging Face Spaces:

1. Creates a non-root user (`appuser`) as required by HF Spaces
2. Installs **CPU-only PyTorch** to save ~1.5 GB image size
3. **Pre-caches** `bert-base-uncased` inside the image to avoid slow cold starts
4. Runs Django migrations and collects static files at build time
5. Serves via Gunicorn on port **7860** (required by HF Spaces)

```bash
# Initialize Git LFS before pushing
git lfs install
git lfs track "models/*.pt"
git add .gitattributes
git commit -m "Track model with LFS"

# Add HF remote and push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/resume-analyzer
git push hf main
```

**Environment variables to set in HF Spaces settings:**

```text
DJANGO_SECRET_KEY=<generate a new key>
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=<your-space-name>.hf.space
```

### Required Environment Variables

| Variable | Description | Required |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | Django secret key (generate fresh per deployment) | ✅ |
| `DJANGO_DEBUG` | `True` for local dev, `False` for production | ✅ |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated list of allowed hostnames | ✅ |
| `HF_TOKEN` | Hugging Face token for private model downloads | Optional |
| `PYTHONDONTWRITEBYTECODE` | Set to `1` for clean Docker images | Optional |
| `PYTHONUNBUFFERED` | Set to `1` for real-time log output | Optional |

---

## 🔧 Standalone ML Usage

Use the prediction pipeline without the web interface:

```python
from src.predict import ResumeJobAnalyzer

# Initialize (loads BERT — first call is slow)
analyzer = ResumeJobAnalyzer()

# Analyze a resume against a job description
result = analyzer.analyze_against_job(
    resume_path="path/to/resume.pdf",
    job_description="We are looking for a Python ML engineer with PyTorch...",
    job_role="ML Engineer"
)

print(f"Match Score:     {result['match_score']}%")
print(f"Matching Skills: {result['matching_skills']}")
print(f"Missing Skills:  {result['missing_skills']}")
print(f"Strengths:       {result['resume_strengths']}")
print(f"Suggestions:     {result['improvement_suggestions']}")
print(f"Predicted Role:  {result['classification']['predicted_role']}")
```

### Batch Candidate Ranking

```python
from src.ranking import CandidateRanker

ranker = CandidateRanker()

jd_text = "Senior ML Engineer role requiring PyTorch, Kubernetes..."
resume_texts = [open(f).read() for f in resume_files]

# Returns top-10 candidates ranked by semantic match
results = ranker.rank_candidates(jd_text, resume_texts, top_k=10)
for r in results:
    print(f"Rank {r['rank']}: Score {r['score']:.1f}%")
```

---

## 📦 Dependencies

```text
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # (required by torch)
transformers>=4.30.0  # BERT tokenizer + model
pdfplumber>=0.9.0     # PDF text extraction
pandas>=2.0.0         # Dataset management
numpy>=1.24.0         # Numerical operations
scikit-learn>=1.3.0   # Metrics (F1, accuracy, confusion matrix)
django>=4.2.0         # Web framework
gunicorn>=21.2.0      # Production WSGI server
whitenoise>=6.5.0     # Static file serving in production
python-dotenv>=1.0.0  # Environment variable loading
tqdm>=4.65.0          # Progress bars
```

---

## 📄 License

This project is for educational and portfolio purposes.

> *Built as a comprehensive deep learning portfolio project demonstrating BERT fine-tuning, transfer learning, robust ML pipeline design, large-scale synthetic data generation, and production-grade web application engineering on Hugging Face Spaces.*
