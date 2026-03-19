"""
Django Forms for Resume Analyzer

Updated to support Resume vs Job Description analysis.
Now accepts: Resume file + Job Role + Job Description text.
"""
from django import forms


class ResumeAnalysisForm(forms.Form):
    """
    Form for analyzing a resume against a job description.
    
    Fields:
        resume_file: Upload resume (PDF or TXT)
        job_role: Target job role (text)
        job_description: Full job description (textarea)
    """
    resume_file = forms.FileField(
        label="Upload Resume",
        help_text="Accepted formats: PDF (.pdf) or Text (.txt)",
        widget=forms.FileInput(attrs={
            "accept": ".pdf,.txt",
            "class": "form-control",
            "id": "resume-file-input"
        })
    )
    
    JOB_ROLE_CHOICES = [
        ("", "— Select a job role —"),
        ("Data Scientist", "Data Scientist"),
        ("Software Engineer", "Software Engineer"),
        ("Web Developer", "Web Developer"),
        ("DevOps Engineer", "DevOps Engineer"),
        ("ML Engineer", "ML Engineer"),
        ("Data Analyst", "Data Analyst"),
        ("Backend Developer", "Backend Developer"),
        ("Frontend Developer", "Frontend Developer"),
        ("Full Stack Developer", "Full Stack Developer"),
        ("Database Administrator", "Database Administrator"),
        ("System Administrator", "System Administrator"),
        ("Network Engineer", "Network Engineer"),
        ("Cybersecurity Analyst", "Cybersecurity Analyst"),
        ("Cloud Architect", "Cloud Architect"),
        ("QA Engineer", "QA Engineer"),
        ("Mobile App Developer", "Mobile App Developer"),
        ("Business Analyst", "Business Analyst"),
        ("IT Project Manager", "IT Project Manager"),
        ("UI UX Designer", "UI UX Designer"),
        ("Data Engineer", "Data Engineer"),
        ("Blockchain Developer", "Blockchain Developer"),
        ("Embedded Systems Engineer", "Embedded Systems Engineer"),
        ("Game Developer", "Game Developer"),
        ("IT Support Specialist", "IT Support Specialist"),
        ("AI Research Scientist", "AI Research Scientist"),
    ]

    job_role = forms.ChoiceField(
        label="Job Role",
        choices=JOB_ROLE_CHOICES,
        help_text="Select the role you're applying for",
        widget=forms.Select(attrs={
            "class": "form-control",
            "id": "job-role-input",
        })
    )
    
    job_description = forms.CharField(
        label="Job Description",
        help_text="Paste the full job description here",
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "id": "job-description-input",
            "rows": 8,
            "placeholder": "Paste the job description text here..."
        })
    )
