"""
Django Views for Resume Analyzer

Handles resume upload + JD input, runs the ML pipeline,
and displays analysis results.
"""
import os
import sys
import json
import tempfile
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect

from .forms import ResumeAnalysisForm

# Add the project root to the path so we can import our ML modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global analyzer instance (loaded once on first request)
_analyzer = None


def get_analyzer():
    """
    Lazy-load the ResumeJobAnalyzer (expensive to initialize).
    
    We load it only once and cache it as a global variable.
    This avoids reloading BERT (~440MB) on every request.
    """
    global _analyzer
    if _analyzer is None:
        from src.predict import ResumeJobAnalyzer
        _analyzer = ResumeJobAnalyzer()
    return _analyzer


def home(request):
    """Home page with resume analysis form."""
    form = ResumeAnalysisForm()
    return render(request, "analyzer/home.html", {"form": form})


@csrf_protect
def analyze(request):
    """
    Handle resume upload + JD and return analysis results.
    
    Security measures:
    - CSRF protection (decorator)
    - File size limit (5 MB)
    - File type validation (extension + magic bytes)
    - Filename sanitization
    - Temp file cleanup in finally block
    """
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

    if request.method != "POST":
        return render(request, "analyzer/home.html", {
            "form": ResumeAnalysisForm()
        })
    
    form = ResumeAnalysisForm(request.POST, request.FILES)
    
    if not form.is_valid():
        return render(request, "analyzer/home.html", {
            "form": form,
            "error": "Please fill in all fields and upload a valid file."
        })
    
    uploaded_file = request.FILES["resume_file"]
    job_role = form.cleaned_data["job_role"]
    job_description = form.cleaned_data["job_description"]
    filename = uploaded_file.name.lower()
    
    # --- Security: File type validation ---
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        return render(request, "analyzer/home.html", {
            "form": ResumeAnalysisForm(),
            "error": "Only PDF and TXT files are supported."
        })
    
    # --- Security: File size validation ---
    if uploaded_file.size > MAX_FILE_SIZE:
        return render(request, "analyzer/home.html", {
            "form": ResumeAnalysisForm(),
            "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB."
        })
    
    # --- Security: Magic bytes validation for PDF ---
    if filename.endswith(".pdf"):
        header = uploaded_file.read(5)
        uploaded_file.seek(0)  # Reset file pointer
        if header != b"%PDF-":
            return render(request, "analyzer/home.html", {
                "form": ResumeAnalysisForm(),
                "error": "Invalid PDF file. The file does not appear to be a valid PDF."
            })
    
    # Save to a temporary file
    suffix = ".pdf" if filename.endswith(".pdf") else ".txt"
    tmp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # Run analysis
        analyzer = get_analyzer()
        result = analyzer.analyze_against_job(
            resume_path=tmp_path,
            job_description=job_description,
            job_role=job_role
        )
        
        return render(request, "analyzer/results.html", {
            "result": result,
            "filename": uploaded_file.name,
            "job_role": job_role
        })
    
    except Exception as e:
        return render(request, "analyzer/home.html", {
            "form": ResumeAnalysisForm(),
            "error": f"Analysis failed: {str(e)}"
        })
    
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
