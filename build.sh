#!/usr/bin/env bash
# =============================================================
# Render Build Script
# =============================================================
# This script runs during Render deployment to:
# 1. Install Python dependencies
# 2. Collect static files
# 3. Run database migrations
# =============================================================

set -o errexit  # Exit on error

echo "============================================================"
echo "  RENDER BUILD"
echo "============================================================"

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Collect static files
echo "[2/4] Collecting static files..."
cd app/django_app
python manage.py collectstatic --noinput

# 3. Run migrations
echo "[3/4] Running migrations..."
python manage.py migrate --noinput

# 4. Download BERT model (cache for faster cold starts)
echo "[4/4] Pre-downloading BERT model..."
python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased'); print('BERT cached!')"

echo "============================================================"
echo "  BUILD COMPLETE"
echo "============================================================"
