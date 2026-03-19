# =============================================================
# Dockerfile — AI Resume Analyzer (Hugging Face Spaces)
# =============================================================
# Deploys the Django app with BERT model on HF Spaces (16GB RAM)
# HF Spaces Docker apps must listen on port 7860
# =============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only PyTorch to save ~1.5GB image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download BERT model (cached in image, fast cold starts)
RUN python -c "\
from transformers import BertTokenizer, BertModel; \
BertTokenizer.from_pretrained('bert-base-uncased'); \
BertModel.from_pretrained('bert-base-uncased'); \
print('BERT model cached!')"

# Copy project files
COPY . .

# Collect Django static files
RUN cd app/django_app && python manage.py collectstatic --noinput 2>/dev/null || true

# Run migrations
RUN cd app/django_app && python manage.py migrate --noinput 2>/dev/null || true

# Fix permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Start gunicorn
CMD ["gunicorn", \
     "--chdir", "app/django_app", \
     "resume_project.wsgi:application", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
