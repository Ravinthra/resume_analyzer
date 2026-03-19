"""
generate_90k_resumes.py
=======================

Large-scale dataset generator for the AI Resume Analyzer project.

Generates:
    90,000 resumes across 25 IT profession classes
    5,000 job descriptions

Dataset Composition
-------------------
Synthetic Resumes : 15,000  (600 per class) — clean, well-structured
Realistic Resumes : 75,000  (3000 per class) — noisy, varied formatting
Job Descriptions  : 5,000   (200 per class)

Key Design: Class-Distinctive Content
--------------------------------------
Each class draws from its OWN unique pool of skills, projects, and
responsibilities (defined in role_data.py). This ensures BERT must learn
real semantic differences to classify resumes.

Cross-class skill mixing (15-25%) adds realism — a Data Scientist
who also knows React, or a DevOps Engineer with SQL experience.

No explicit job titles appear in resume text to prevent label leakage.
"""

import csv
import json
import random
from pathlib import Path

from role_data import ROLE_DATA

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

NUM_CLASSES     = 25
SYNTH_PER_CLASS = 600
REAL_PER_CLASS  = 3000
JD_PER_CLASS    = 200

PROJECT_ROOT = Path(__file__).parent.parent

SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic_resumes"
REAL_DIR  = PROJECT_ROOT / "data" / "real_resumes"
JD_DIR    = PROJECT_ROOT / "data" / "job_descriptions"
META_DIR  = PROJECT_ROOT / "data" / "metadata"

random.seed(42)

# ---------------------------------------------------------
# ROLE LABELS (must match ROLE_DATA keys exactly)
# ---------------------------------------------------------

ROLE_NAMES = list(ROLE_DATA.keys())
assert len(ROLE_NAMES) == NUM_CLASSES, f"Expected {NUM_CLASSES} roles, got {len(ROLE_NAMES)}"

# Collect ALL skills across all roles for cross-class mixing
ALL_SKILLS = []
for role_info in ROLE_DATA.values():
    ALL_SKILLS.extend(role_info["skills"])
ALL_SKILLS = list(set(ALL_SKILLS))  # deduplicate


# ---------------------------------------------------------
# NAME & EMAIL POOLS
# ---------------------------------------------------------

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Chris", "Avery", "Sam", "Riley",
    "Casey", "Jamie", "Quinn", "Dakota", "Skyler", "Reese", "Cameron",
    "Drew", "Rowan", "Blake", "Hayden", "Emerson", "Finley", "Parker",
    "Sage", "River", "Phoenix", "Arin", "Kai", "Noel", "Remy", "Sasha",
]

LAST_NAMES = [
    "Smith", "Patel", "Chen", "Garcia", "Kumar", "Brown", "Lee", "Singh",
    "Johnson", "Williams", "Davis", "Rodriguez", "Martinez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Thompson",
    "Nguyen", "Kim", "Tanaka", "Mueller", "Ivanov", "Johansson",
]

EDUCATION = [
    "B.S. Computer Science", "M.S. Computer Science",
    "B.Tech Information Technology", "M.S. Data Science",
    "B.S. Software Engineering", "B.S. Information Systems",
    "M.S. Artificial Intelligence", "B.S. Electrical Engineering",
    "M.S. Cybersecurity", "B.S. Mathematics",
    "B.Eng Computer Engineering", "M.S. Information Technology",
]

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "Georgia Tech", "IIT Bombay",
    "University of Toronto", "ETH Zurich", "University of Michigan",
    "Purdue University", "University of Texas Austin", "UCLA",
    "Columbia University", "University of Washington", "NTU Singapore",
]

DOMAINS = ["gmail.com", "outlook.com", "protonmail.com", "yahoo.com", "mail.com"]


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------

def rand_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_email(name: str) -> str:
    tag = random.randint(10, 999)
    return f"{name.lower().replace(' ', '.')}{tag}@{random.choice(DOMAINS)}"


def rand_years() -> str:
    return f"{random.randint(1, 15)} years"


def rand_company() -> str:
    prefixes = ["Tech", "Data", "Cloud", "Next", "Core", "Prime", "Alpha", "Nexus", "Apex", "Vertex"]
    suffixes = ["Corp", "Systems", "Labs", "Solutions", "Industries", "Group", "Digital", "Works"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"


# ---------------------------------------------------------
# RESUME GENERATORS
# ---------------------------------------------------------

def generate_synthetic_resume(role: str) -> str:
    """
    Clean, well-structured resume.
    Uses 75-85% class-specific skills + 15-25% cross-class skills.
    No explicit job title in the text.
    """
    role_info = ROLE_DATA[role]
    name = rand_name()

    # Skills: majority from this class, some from other classes
    n_class_skills = random.randint(8, 12)
    n_cross_skills = random.randint(2, 4)

    class_skills = random.sample(role_info["skills"], min(n_class_skills, len(role_info["skills"])))

    # Cross-class skills (exclude this role's skills)
    other_skills = [s for s in ALL_SKILLS if s not in role_info["skills"]]
    cross_skills = random.sample(other_skills, min(n_cross_skills, len(other_skills)))

    all_skills = class_skills + cross_skills
    random.shuffle(all_skills)

    # Projects: 2-3 from this class
    projects = random.sample(role_info["projects"], min(3, len(role_info["projects"])))

    # Responsibilities: 2-3 from this class
    resps = random.sample(role_info["responsibilities"], min(3, len(role_info["responsibilities"])))

    resume = f"""{name}
Email: {rand_email(name)}

SUMMARY
{role_info['summary']}. {rand_years()} of professional experience.

SKILLS
{', '.join(all_skills)}

EXPERIENCE

{rand_company()} | {rand_years()}
- {resps[0]}
- {resps[1]}
{f'- {resps[2]}' if len(resps) > 2 else ''}

{rand_company()} | {rand_years()}
- {resps[-1]}

PROJECTS
- {projects[0]}
- {projects[1]}
{f'- {projects[2]}' if len(projects) > 2 else ''}

EDUCATION
{random.choice(EDUCATION)}, {random.choice(UNIVERSITIES)}
"""
    return resume.strip()


def generate_realistic_resume(role: str) -> str:
    """
    Noisy, varied-format resume simulating real-world messiness.
    Includes: random formatting, varied section ordering, incomplete sections,
    abbreviations, extra whitespace, etc.
    Still uses 70-80% class-specific skills for learnability.
    """
    role_info = ROLE_DATA[role]
    name = rand_name()

    # Skills: more cross-class noise than synthetic
    n_class_skills = random.randint(6, 10)
    n_cross_skills = random.randint(3, 6)

    class_skills = random.sample(role_info["skills"], min(n_class_skills, len(role_info["skills"])))
    other_skills = [s for s in ALL_SKILLS if s not in role_info["skills"]]
    cross_skills = random.sample(other_skills, min(n_cross_skills, len(other_skills)))

    all_skills = class_skills + cross_skills
    random.shuffle(all_skills)

    projects = random.sample(role_info["projects"], min(random.randint(1, 3), len(role_info["projects"])))
    resps = random.sample(role_info["responsibilities"], min(random.randint(2, 4), len(role_info["responsibilities"])))

    # Varied formatting styles
    style = random.choice(["formal", "minimal", "detailed", "bullet_heavy"])

    if style == "formal":
        resume = f"""{name.upper()}
{rand_email(name)} | linkedin.com/in/{name.lower().replace(' ', '-')}

Professional Summary:
{role_info['summary']}

Technical Skills:
{' | '.join(all_skills)}

Work Experience:

{rand_company()}
{rand_years()} of experience
{chr(10).join('* ' + r for r in resps)}

Key Projects:
{chr(10).join('> ' + p for p in projects)}

Education:
{random.choice(EDUCATION)} - {random.choice(UNIVERSITIES)}
"""

    elif style == "minimal":
        resume = f"""{name}
{rand_email(name)}

skills: {', '.join(all_skills)}

experience:
{chr(10).join('- ' + r for r in resps[:2])}

projects:
{chr(10).join('- ' + p for p in projects[:2])}

education: {random.choice(EDUCATION)}
"""

    elif style == "detailed":
        company1 = rand_company()
        company2 = rand_company()
        resume = f"""{name}
Contact: {rand_email(name)}

ABOUT ME
{role_info['summary']}. Passionate about delivering high-quality solutions with {rand_years()} of hands-on experience.

CORE COMPETENCIES
{chr(10).join('  - ' + s for s in all_skills)}

PROFESSIONAL EXPERIENCE

{company1} ({rand_years()})
Role highlights:
{chr(10).join('  * ' + r for r in resps)}

{company2} ({rand_years()})
Role highlights:
  * {resps[0]}

NOTABLE PROJECTS
{chr(10).join('  - ' + p for p in projects)}

EDUCATION
{random.choice(EDUCATION)}
{random.choice(UNIVERSITIES)}

CERTIFICATIONS
{random.choice(['AWS Certified', 'Azure Certified', 'Google Cloud Certified', 'PMP', 'CISSP', 'CKA', 'Scrum Master', 'CCNA'])}
"""

    else:  # bullet_heavy
        resume = f"""{name}
{rand_email(name)}

• {role_info['summary']}

• Skills: {', '.join(all_skills)}

• Experience ({rand_years()}):
{chr(10).join('  • ' + r for r in resps)}

• Projects:
{chr(10).join('  • ' + p for p in projects)}

• Education: {random.choice(EDUCATION)}, {random.choice(UNIVERSITIES)}
"""

    # Add random noise
    if random.random() < 0.3:
        resume = resume.replace(",", " ,")  # bad spacing
    if random.random() < 0.2:
        resume = resume.replace("\n\n", "\n\n\n")  # extra whitespace
    if random.random() < 0.15:
        lines = resume.split("\n")
        random.shuffle(lines[-5:])  # shuffle some bottom lines
        resume = "\n".join(lines)

    return resume.strip()


def generate_jd(role: str) -> str:
    """Generate a job description for a specific role."""
    role_info = ROLE_DATA[role]
    skills = random.sample(role_info["skills"], min(8, len(role_info["skills"])))
    resps = random.sample(role_info["responsibilities"], min(3, len(role_info["responsibilities"])))

    return f"""Job Title: {role}

About the Role:
We are looking for an experienced professional to join our team.
{role_info['summary']}.

Required Skills:
{chr(10).join('- ' + s for s in skills)}

Responsibilities:
{chr(10).join('- ' + r for r in resps)}

Qualifications:
- {random.choice(EDUCATION)} or equivalent
- {random.randint(2, 8)}+ years of relevant experience
"""


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    import time
    start = time.time()

    META_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    total_files = 0

    for class_id, role in enumerate(ROLE_NAMES):
        slug = role.lower().replace(" ", "_").replace("/", "_")

        synth_dir = SYNTH_DIR / slug
        real_dir  = REAL_DIR / slug
        jd_dir    = JD_DIR / slug

        synth_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)
        jd_dir.mkdir(parents=True, exist_ok=True)

        # Synthetic resumes
        for i in range(SYNTH_PER_CLASS):
            text = generate_synthetic_resume(role)
            path = synth_dir / f"resume_{i:04d}.txt"
            path.write_text(text, encoding="utf-8")
            entries.append({
                "resume_path": str(path.relative_to(PROJECT_ROOT)),
                "label": role,
                "label_id": class_id,
                "source": "synthetic",
            })

        # Realistic resumes
        for i in range(REAL_PER_CLASS):
            text = generate_realistic_resume(role)
            path = real_dir / f"resume_{i:04d}.txt"
            path.write_text(text, encoding="utf-8")
            entries.append({
                "resume_path": str(path.relative_to(PROJECT_ROOT)),
                "label": role,
                "label_id": class_id,
                "source": "real",
            })

        # Job descriptions
        for i in range(JD_PER_CLASS):
            jd = generate_jd(role)
            path = jd_dir / f"jd_{i:04d}.txt"
            path.write_text(jd, encoding="utf-8")

        total_files += SYNTH_PER_CLASS + REAL_PER_CLASS + JD_PER_CLASS
        elapsed = time.time() - start
        print(f"  [{class_id+1:2d}/{NUM_CLASSES}] {role:<30s} — {SYNTH_PER_CLASS + REAL_PER_CLASS + JD_PER_CLASS} files  ({elapsed:.1f}s)")

    # Shuffle and save CSV
    random.shuffle(entries)

    csv_path = META_DIR / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_path", "label", "label_id", "source"])
        writer.writeheader()
        writer.writerows(entries)

    # Save labels JSON
    labels_dict = {
        "num_classes": NUM_CLASSES,
        "label_to_id": {r: i for i, r in enumerate(ROLE_NAMES)},
        "id_to_label": {str(i): r for i, r in enumerate(ROLE_NAMES)},
    }
    with open(META_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=4)

    elapsed = time.time() - start
    print(f"\nDataset generation complete in {elapsed:.1f}s")
    print(f"Total resumes  : {len(entries):,}")
    print(f"Total files    : {total_files:,}")
    print(f"CSV saved to   : {csv_path}")


if __name__ == "__main__":
    main()