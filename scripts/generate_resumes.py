"""
generate_resumes.py -- Synthetic Resume Generator
====================================================

Generates 120 diverse, realistic resumes per job role (600 total)
for training the BERT classifier with production-quality data.

Each resume varies in:
- Candidate name, education, university, GPA
- Years of experience (1-15)
- Companies (from curated pools)
- Projects and descriptions
- Skill subsets (from role-specific pools)
- Resume writing style and format
- Certifications and achievements

Usage:
    python scripts/generate_resumes.py
"""

import os
import csv
import json
import random
from pathlib import Path

# ===========================================================================
# CONFIGURATION
# ===========================================================================

RESUMES_PER_CLASS = 120
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "resumes"
CSV_PATH = PROJECT_ROOT / "data" / "dataset.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "labels.json"

random.seed(42)

# ===========================================================================
# NAME POOLS (diverse backgrounds)
# ===========================================================================

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna",
    "Arun", "Priya", "Raj", "Ananya", "Vikram", "Neha", "Suresh", "Pooja",
    "Wei", "Mei", "Hiroshi", "Yuki", "Jun", "Sakura", "Takeshi", "Ai",
    "Carlos", "Maria", "Pedro", "Ana", "Miguel", "Isabella", "Diego", "Sofia",
    "Ahmed", "Fatima", "Omar", "Layla", "Ali", "Noor", "Hassan", "Amira",
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Sanjay", "Divya", "Amit", "Shreya", "Ravi", "Kavitha", "Deepak", "Lakshmi",
    "Yusuf", "Zara", "Ibrahim", "Hana", "Tariq", "Salma", "Khalid", "Rana",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Patel", "Kumar", "Singh", "Sharma", "Gupta", "Reddy", "Nair", "Verma",
    "Chen", "Wang", "Li", "Zhang", "Liu", "Yang", "Huang", "Wu",
    "Tanaka", "Yamamoto", "Sato", "Suzuki", "Nakamura", "Watanabe", "Kobayashi",
    "Kim", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Jang",
    "Al-Hassan", "Ben-David", "O'Brien", "Van der Berg", "De Silva", "Nakamura",
    "Fernandez", "Costa", "Santos", "Oliveira", "Souza", "Almeida", "Ribeiro",
]

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "Georgia Tech", "University of Illinois",
    "University of Michigan", "University of Washington",
    "Columbia University", "Cornell University",
    "University of Texas at Austin", "Purdue University",
    "University of Wisconsin", "Penn State University",
    "University of Maryland", "Virginia Tech",
    "IIT Bombay", "IIT Delhi", "IIT Madras", "BITS Pilani",
    "NUS Singapore", "Tsinghua University", "University of Toronto",
    "ETH Zurich", "University of Oxford", "Imperial College London",
    "KAIST", "University of Melbourne", "TU Munich",
    "University of Waterloo", "McGill University",
    "National University of Singapore", "Peking University",
]

DEGREES = [
    "B.S. in Computer Science", "M.S. in Computer Science",
    "B.S. in Data Science", "M.S. in Data Science",
    "B.S. in Software Engineering", "M.S. in Software Engineering",
    "B.Tech in Computer Science", "M.Tech in Computer Science",
    "B.S. in Information Technology", "M.S. in Information Technology",
    "B.S. in Mathematics", "M.S. in Applied Mathematics",
    "B.S. in Statistics", "M.S. in Statistics",
    "B.S. in Electrical Engineering", "M.S. in Machine Learning",
    "Ph.D. in Computer Science", "Ph.D. in Machine Learning",
    "MBA in Technology Management", "B.S. in Physics",
]

TECH_COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
    "Uber", "Airbnb", "Stripe", "Shopify", "Salesforce", "Adobe",
    "Twitter", "LinkedIn", "Snap", "Pinterest", "Spotify", "Slack",
    "Palantir", "Databricks", "Snowflake", "Confluent", "HashiCorp",
    "IBM", "Oracle", "SAP", "VMware", "Cisco", "Intel", "NVIDIA",
    "Goldman Sachs", "JPMorgan", "Morgan Stanley", "Bloomberg",
    "Accenture", "Deloitte", "McKinsey", "Infosys", "TCS", "Wipro",
    "Twilio", "Atlassian", "Zoom", "Cloudflare", "Okta",
    "DataRobot", "H2O.ai", "Weights & Biases", "Scale AI", "OpenAI",
    "ByteDance", "Tencent", "Alibaba", "Baidu", "Samsung",
    "Startup (Series A)", "Startup (Series B)", "Startup (Seed Stage)",
]

# ===========================================================================
# ROLE-SPECIFIC DATA
# ===========================================================================

ROLE_CONFIG = {
    "Data Scientist": {
        "id": 0,
        "skills_pool": [
            "Python", "R", "SQL", "TensorFlow", "PyTorch", "Scikit-learn",
            "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
            "Jupyter Notebook", "Spark", "Hadoop", "Hive",
            "Statistical Modeling", "A/B Testing", "Hypothesis Testing",
            "Linear Regression", "Logistic Regression", "Random Forests",
            "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
            "Neural Networks", "Deep Learning", "NLP", "Computer Vision",
            "Time Series Analysis", "Bayesian Statistics", "MCMC",
            "Feature Engineering", "Dimensionality Reduction", "PCA",
            "Clustering", "K-Means", "DBSCAN", "SVM",
            "Data Visualization", "Tableau", "Power BI", "Looker",
            "AWS SageMaker", "Google BigQuery", "Azure ML",
            "Git", "Docker", "Airflow", "MLflow", "DVC",
            "Data Wrangling", "ETL", "Data Pipeline",
            "Keras", "NLTK", "spaCy", "Hugging Face",
        ],
        "projects": [
            "Built a customer churn prediction model using gradient boosting, achieving 92% accuracy on a dataset of 500K customers",
            "Developed a recommendation system using collaborative filtering and matrix factorization for e-commerce platform serving 10M users",
            "Created a time series forecasting pipeline for demand prediction using ARIMA and LSTM models, reducing inventory costs by 15%",
            "Implemented an NLP pipeline for sentiment analysis of customer reviews using BERT, processing 1M reviews daily",
            "Designed and executed A/B tests across 50+ experiments, driving $2M in incremental revenue",
            "Built a fraud detection system using anomaly detection and ensemble methods, identifying 95% of fraudulent transactions",
            "Developed a customer segmentation model using K-means clustering and RFM analysis on 2M customer records",
            "Created a pricing optimization model using reinforcement learning, improving margins by 8%",
            "Built an image classification pipeline using CNNs for defect detection in manufacturing, achieving 98% accuracy",
            "Designed a real-time dashboard with Tableau for executive reporting across 15 KPIs",
            "Implemented a feature store using Feast for ML model serving across 20+ models in production",
            "Developed a causal inference framework using DoWhy for marketing attribution modeling",
            "Built a text classification system for support ticket routing using TF-IDF and logistic regression",
            "Created a survival analysis model for customer lifetime value prediction using Cox proportional hazards",
            "Developed a geospatial analysis pipeline for store location optimization using spatial clustering",
        ],
        "responsibilities": [
            "Led data science initiatives across product and engineering teams",
            "Conducted exploratory data analysis on large-scale datasets",
            "Built and deployed machine learning models to production",
            "Designed experiments and analyzed results for product decisions",
            "Mentored junior data scientists on statistical methods",
            "Presented insights to C-level executives and stakeholders",
            "Collaborated with data engineers to build scalable data pipelines",
            "Developed automated reporting and monitoring dashboards",
            "Performed statistical analysis to drive business strategy",
            "Established best practices for model development lifecycle",
        ],
        "certifications": [
            "AWS Certified Machine Learning - Specialty",
            "Google Professional Data Engineer",
            "IBM Data Science Professional Certificate",
            "Coursera Deep Learning Specialization",
            "Microsoft Certified: Azure Data Scientist Associate",
            "SAS Certified Data Scientist",
        ],
    },

    "Software Engineer": {
        "id": 1,
        "skills_pool": [
            "Python", "Java", "C++", "C#", "Go", "Rust", "TypeScript", "JavaScript",
            "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "REST APIs", "GraphQL", "gRPC", "Microservices", "Event-Driven Architecture",
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
            "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",
            "Git", "Linux", "Bash", "Shell Scripting",
            "Data Structures", "Algorithms", "System Design",
            "Object-Oriented Programming", "Design Patterns", "SOLID Principles",
            "Unit Testing", "Integration Testing", "TDD", "BDD",
            "Agile", "Scrum", "Kanban", "JIRA",
            "Spring Boot", "Django", "Flask", "FastAPI",
            "Message Queues", "Kafka", "RabbitMQ", "SQS",
            "Distributed Systems", "Load Balancing", "Caching",
            "Monitoring", "Prometheus", "Grafana", "Datadog",
            "Security Best Practices", "OAuth", "JWT",
            "HTTP", "TCP/IP", "WebSockets", "Protocol Buffers",
        ],
        "projects": [
            "Designed and built a microservices architecture handling 100K requests per second with 99.99% uptime",
            "Developed a real-time payment processing system handling $50M in daily transactions",
            "Built a distributed task queue system using Kafka and Redis, processing 5M events per hour",
            "Migrated monolithic application to microservices, reducing deployment time from 2 hours to 5 minutes",
            "Implemented a caching layer using Redis, reducing API latency by 60% and database load by 40%",
            "Designed a multi-tenant SaaS platform serving 500+ enterprise customers",
            "Built an automated CI/CD pipeline with GitHub Actions, reducing release cycles from weekly to daily",
            "Developed a search service using Elasticsearch supporting full-text search across 100M documents",
            "Created a rate limiting and throttling system protecting APIs from abuse at scale",
            "Built a notification service delivering 10M push notifications daily across iOS and Android",
            "Implemented an event sourcing system for audit logging and data recovery",
            "Designed database sharding strategy handling 1TB+ of data with sub-100ms query times",
            "Developed a feature flagging system enabling safe rollouts for 50+ engineering teams",
            "Built a GraphQL API gateway aggregating data from 15 backend services",
            "Created an internal developer platform reducing onboarding time from 2 weeks to 2 days",
        ],
        "responsibilities": [
            "Designed and implemented scalable backend services",
            "Led code reviews and established coding standards",
            "Collaborated with product managers on technical requirements",
            "Optimized system performance and reduced latency",
            "Mentored junior engineers through pair programming",
            "Participated in on-call rotation for production systems",
            "Wrote technical documentation and architecture decision records",
            "Contributed to open-source projects and internal tools",
            "Conducted system design reviews for new features",
            "Drove technical decisions and architecture discussions",
        ],
        "certifications": [
            "AWS Certified Solutions Architect - Professional",
            "Google Cloud Professional Cloud Architect",
            "Certified Kubernetes Administrator (CKA)",
            "Oracle Certified Professional, Java SE",
            "Microsoft Certified: Azure Developer Associate",
            "HashiCorp Certified: Terraform Associate",
        ],
    },

    "Web Developer": {
        "id": 2,
        "skills_pool": [
            "HTML5", "CSS3", "JavaScript", "TypeScript",
            "React", "Angular", "Vue.js", "Next.js", "Svelte", "Nuxt.js",
            "Node.js", "Express.js", "Nest.js", "Fastify",
            "Python", "Django", "Flask", "Ruby on Rails",
            "PHP", "Laravel", "WordPress",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "Firebase",
            "REST APIs", "GraphQL", "WebSockets",
            "Tailwind CSS", "Bootstrap", "Material UI", "Styled Components", "SASS",
            "Webpack", "Vite", "Babel", "ESLint", "Prettier",
            "Jest", "Cypress", "Playwright", "React Testing Library",
            "Git", "GitHub", "GitLab",
            "AWS", "Vercel", "Netlify", "Heroku", "DigitalOcean",
            "Docker", "Nginx", "Apache",
            "Responsive Design", "Mobile First", "Progressive Web Apps",
            "SEO Optimization", "Web Accessibility", "WCAG",
            "Figma", "Adobe XD", "Sketch",
            "State Management", "Redux", "Zustand", "MobX",
            "Authentication", "OAuth", "JWT", "Auth0",
            "Performance Optimization", "Lighthouse", "Core Web Vitals",
        ],
        "projects": [
            "Built a progressive web app for an e-commerce platform with 50K daily active users using React and Next.js",
            "Developed a real-time collaborative document editor using WebSockets and operational transforms",
            "Created a responsive dashboard with data visualization using D3.js and React, serving 200+ enterprise clients",
            "Built a headless CMS-powered marketing website achieving 95+ Lighthouse scores across all metrics",
            "Developed a social media platform with React Native for cross-platform mobile support",
            "Implemented a design system library with 50+ reusable components used across 10 product teams",
            "Built an interactive data visualization platform using D3.js processing 1M data points",
            "Created a multi-language e-learning platform with video streaming and progress tracking",
            "Developed a booking and scheduling system with calendar integration and payment processing",
            "Built a portfolio management dashboard with real-time stock data and charting using Chart.js",
            "Implemented server-side rendering with Next.js, improving SEO rankings by 40%",
            "Created an accessible web application meeting WCAG 2.1 AA standards across all pages",
            "Built a drag-and-drop website builder using React DnD and a custom rendering engine",
            "Developed a real-time chat application with typing indicators using Socket.io",
            "Created a PWA with offline support and push notifications for a delivery service",
        ],
        "responsibilities": [
            "Developed and maintained responsive web applications",
            "Collaborated with UX designers to implement pixel-perfect interfaces",
            "Optimized web performance and Core Web Vitals scores",
            "Implemented SEO best practices and accessibility standards",
            "Conducted cross-browser testing and bug fixes",
            "Managed frontend build pipelines and deployment",
            "Led frontend architecture decisions and tech stack selection",
            "Integrated third-party APIs and payment gateways",
            "Implemented state management solutions for complex UIs",
            "Wrote unit and integration tests for frontend components",
        ],
        "certifications": [
            "Meta Front-End Developer Professional Certificate",
            "AWS Certified Cloud Practitioner",
            "Google UX Design Professional Certificate",
            "freeCodeCamp Full Stack Certification",
            "Udacity Full Stack Web Developer Nanodegree",
            "W3C Web Accessibility Specialist",
        ],
    },

    "DevOps Engineer": {
        "id": 3,
        "skills_pool": [
            "Linux", "Ubuntu", "CentOS", "RHEL", "Bash", "Shell Scripting",
            "Python", "Go", "Ruby",
            "AWS", "GCP", "Azure", "Multi-Cloud Architecture",
            "Docker", "Kubernetes", "Helm", "Istio", "Containerd",
            "Terraform", "Ansible", "Puppet", "Chef", "CloudFormation",
            "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "ArgoCD",
            "Git", "GitOps", "Trunk-Based Development",
            "Prometheus", "Grafana", "Datadog", "New Relic", "ELK Stack",
            "Splunk", "PagerDuty", "OpsGenie",
            "Nginx", "HAProxy", "Traefik", "Load Balancing",
            "PostgreSQL", "MySQL", "Redis", "MongoDB",
            "Kafka", "RabbitMQ", "SQS", "SNS",
            "Networking", "TCP/IP", "DNS", "VPN", "Firewalls",
            "Security", "IAM", "SSL/TLS", "Vault", "Secrets Management",
            "Infrastructure as Code", "Configuration Management",
            "Monitoring", "Alerting", "Incident Management",
            "SRE Practices", "SLOs", "SLIs", "Error Budgets",
            "Cost Optimization", "FinOps", "Auto-Scaling",
            "Disaster Recovery", "High Availability", "Backup Strategies",
        ],
        "projects": [
            "Designed and implemented a Kubernetes cluster managing 500+ microservices across 3 availability zones",
            "Built a CI/CD pipeline reducing deployment time from 4 hours to 15 minutes using ArgoCD and GitHub Actions",
            "Migrated on-premises infrastructure to AWS, reducing operational costs by 40% and improving uptime to 99.99%",
            "Implemented infrastructure as code using Terraform, managing 200+ cloud resources across 5 environments",
            "Designed a multi-region disaster recovery strategy with RPO < 1 minute and RTO < 5 minutes",
            "Built a centralized logging and monitoring platform using ELK Stack, processing 50GB of logs daily",
            "Implemented auto-scaling policies that reduced cloud costs by $500K annually while maintaining SLAs",
            "Created a self-service platform for developers to provision environments in under 10 minutes",
            "Designed and implemented zero-downtime deployment strategy for 100+ services",
            "Built a secrets management system using HashiCorp Vault serving 200+ applications",
            "Implemented container security scanning in CI/CD pipeline, catching 95% of vulnerabilities pre-deployment",
            "Designed a service mesh using Istio for inter-service communication with mTLS encryption",
            "Built automated database backup and recovery system with point-in-time restore capability",
            "Created a chaos engineering framework for testing system resilience under failure conditions",
            "Implemented FinOps practices reducing monthly cloud spend by 30% through right-sizing and reserved instances",
        ],
        "responsibilities": [
            "Managed and maintained cloud infrastructure across multiple environments",
            "Designed and implemented CI/CD pipelines for development teams",
            "Monitored system health and resolved production incidents",
            "Automated infrastructure provisioning and configuration",
            "Implemented security best practices and compliance requirements",
            "Collaborated with development teams on deployment strategies",
            "Conducted capacity planning and cost optimization",
            "Led incident response and post-mortem processes",
            "Maintained documentation for infrastructure and runbooks",
            "Evaluated and adopted new DevOps tools and practices",
        ],
        "certifications": [
            "AWS Certified DevOps Engineer - Professional",
            "Certified Kubernetes Administrator (CKA)",
            "HashiCorp Certified: Terraform Associate",
            "Google Cloud Professional DevOps Engineer",
            "Red Hat Certified System Administrator",
            "Linux Foundation Certified System Administrator",
        ],
    },

    "ML Engineer": {
        "id": 4,
        "skills_pool": [
            "Python", "C++", "Java", "Scala",
            "PyTorch", "TensorFlow", "JAX", "Keras",
            "Scikit-learn", "XGBoost", "LightGBM",
            "Hugging Face Transformers", "ONNX", "TensorRT",
            "NLP", "Computer Vision", "Speech Recognition", "Reinforcement Learning",
            "BERT", "GPT", "T5", "CLIP", "Stable Diffusion", "LLMs",
            "CNNs", "RNNs", "LSTMs", "Transformers", "GANs", "VAEs",
            "MLflow", "Kubeflow", "MLOps", "Model Serving",
            "Docker", "Kubernetes", "AWS SageMaker", "Vertex AI",
            "Feature Engineering", "Data Pipelines", "Airflow",
            "Distributed Training", "Model Parallelism", "Data Parallelism",
            "Model Optimization", "Quantization", "Pruning", "Distillation",
            "CUDA", "GPU Programming", "Mixed Precision Training",
            "Git", "DVC", "Weights & Biases", "Neptune",
            "A/B Testing", "Statistical Analysis", "Experiment Tracking",
            "REST APIs", "FastAPI", "Flask", "gRPC",
            "SQL", "PostgreSQL", "MongoDB", "Redis",
            "Spark", "Ray", "Dask", "Apache Beam",
            "Linux", "Bash", "CI/CD",
        ],
        "projects": [
            "Built an end-to-end ML pipeline for real-time fraud detection processing 10K transactions per second with 99.7% accuracy",
            "Deployed a BERT-based NLP model for intent classification serving 5M API requests daily with < 50ms latency",
            "Developed a computer vision pipeline for autonomous vehicle object detection using YOLOv5 with 95% mAP",
            "Built a recommendation engine using collaborative filtering and deep learning, increasing user engagement by 25%",
            "Implemented a model serving infrastructure using TensorFlow Serving and Kubernetes, handling 100K inferences/sec",
            "Fine-tuned GPT models for domain-specific text generation, reducing content creation costs by 60%",
            "Designed an A/B testing framework for ML model evaluation with automated statistical significance testing",
            "Built a real-time speech-to-text system using Whisper with custom domain adaptation for medical terminology",
            "Developed a model monitoring system tracking data drift and performance degradation across 50+ production models",
            "Created an automated ML pipeline using Kubeflow, reducing model deployment time from 2 weeks to 2 hours",
            "Implemented distributed training on 8 GPUs using PyTorch DDP, reducing training time by 6x",
            "Built a feature store using Feast for online and offline feature serving across 30+ ML models",
            "Developed a text summarization system using T5 for processing 100K documents daily",
            "Created a model compression pipeline using knowledge distillation, reducing model size by 4x with < 2% accuracy loss",
            "Built an image generation pipeline using Stable Diffusion for automated product photography",
        ],
        "responsibilities": [
            "Designed and implemented production ML systems end-to-end",
            "Optimized model inference latency and throughput",
            "Built and maintained ML infrastructure and pipelines",
            "Collaborated with data scientists to productionize research models",
            "Implemented model monitoring and alerting systems",
            "Conducted experiments and hyperparameter optimization",
            "Developed APIs for model serving and integration",
            "Managed GPU clusters and training infrastructure",
            "Led MLOps practices and model lifecycle management",
            "Stayed current with state-of-the-art ML research and techniques",
        ],
        "certifications": [
            "AWS Certified Machine Learning - Specialty",
            "Google Professional Machine Learning Engineer",
            "NVIDIA Deep Learning Institute Certification",
            "TensorFlow Developer Certificate",
            "Coursera Machine Learning Engineering for Production",
            "Microsoft Certified: Azure AI Engineer Associate",
        ],
    },
}

# ===========================================================================
# RESUME TEMPLATES (varied formats)
# ===========================================================================

RESUME_TEMPLATES = [
    # Template 1: Traditional format
    """{name}
{email}
Phone: {phone}
LinkedIn: linkedin.com/in/{linkedin}
GitHub: github.com/{github}

SUMMARY
{summary}

EDUCATION
{degree}
{university}
GPA: {gpa}/4.0 | Graduated: {grad_year}

EXPERIENCE

{role_title} | {company1} | {date1}
{responsibilities1}

{prev_title} | {company2} | {date2}
{responsibilities2}

PROJECTS
{projects}

SKILLS
{skills}

CERTIFICATIONS
{certifications}""",

    # Template 2: Modern concise
    """{name}
{email} | {phone} | linkedin.com/in/{linkedin}

--- PROFESSIONAL SUMMARY ---
{summary}

--- SKILLS ---
{skills}

--- EXPERIENCE ---

{role_title}, {company1}
{date1}
{responsibilities1}

{prev_title}, {company2}
{date2}
{responsibilities2}

--- PROJECTS ---
{projects}

--- EDUCATION ---
{degree}, {university} ({grad_year})
GPA: {gpa}

--- CERTIFICATIONS ---
{certifications}""",

    # Template 3: Skills-first
    """{name}
Contact: {email} | {phone}
Portfolio: github.com/{github}

TECHNICAL SKILLS
{skills}

PROFESSIONAL EXPERIENCE

{company1} - {role_title}
{date1}
{responsibilities1}

{company2} - {prev_title}
{date2}
{responsibilities2}

KEY PROJECTS
{projects}

EDUCATION
{university}
{degree} | {grad_year} | GPA: {gpa}/4.0

AWARDS & CERTIFICATIONS
{certifications}""",

    # Template 4: Narrative style
    """{name}
{email}
{phone}

ABOUT ME
{summary}

WHAT I BRING
{skills}

WHERE I HAVE WORKED

At {company1}, I served as a {role_title} ({date1}).
{responsibilities1}

Previously at {company2}, I worked as a {prev_title} ({date2}).
{responsibilities2}

NOTABLE PROJECTS
{projects}

ACADEMIC BACKGROUND
{degree} from {university}, graduated {grad_year} with a {gpa} GPA.

CREDENTIALS
{certifications}""",
]

# ===========================================================================
# GENERATION FUNCTIONS
# ===========================================================================

def random_phone():
    return f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}"

def random_email(first, last):
    domains = ["gmail.com", "outlook.com", "yahoo.com", "protonmail.com", "icloud.com"]
    separators = [".", "_", ""]
    sep = random.choice(separators)
    num = random.choice(["", str(random.randint(1,99))])
    return f"{first.lower()}{sep}{last.lower()}{num}@{random.choice(domains)}"

def random_date_range(years_exp):
    end_year = random.choice([2024, 2025, 2026])
    start_year = end_year - random.randint(1, min(years_exp, 4))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{random.choice(months)} {start_year} - {random.choice(months)} {end_year}"

def random_prev_date(years_exp):
    end_year = random.randint(2019, 2023)
    start_year = end_year - random.randint(1, min(years_exp, 3))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{random.choice(months)} {start_year} - {random.choice(months)} {end_year}"

def generate_summary(role_name, years_exp, skills_sample):
    templates = [
        f"Experienced {role_name} with {years_exp}+ years of expertise in {', '.join(skills_sample[:3])}. Passionate about building scalable solutions and driving business impact through technology.",
        f"Results-driven {role_name} with {years_exp} years of experience specializing in {', '.join(skills_sample[:3])}. Strong track record of delivering high-quality projects on schedule.",
        f"Dedicated {role_name} bringing {years_exp} years of hands-on experience with {', '.join(skills_sample[:4])}. Known for innovative problem-solving and collaborative teamwork.",
        f"{role_name} with a proven {years_exp}-year track record in {', '.join(skills_sample[:3])} and a passion for continuous learning and technical excellence.",
        f"Detail-oriented {role_name} with {years_exp}+ years building production systems using {', '.join(skills_sample[:3])}. Experienced in both startup and enterprise environments.",
    ]
    return random.choice(templates)

def generate_resume(role_name, config, index):
    """Generate a single synthetic resume."""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    years_exp = random.randint(1, 15)

    # Select random subset of skills (8-18 skills)
    num_skills = random.randint(8, 18)
    skills_sample = random.sample(config["skills_pool"], min(num_skills, len(config["skills_pool"])))

    # Pick companies
    companies = random.sample(TECH_COMPANIES, 2)

    # Pick projects (2-4)
    num_projects = random.randint(2, 4)
    projects = random.sample(config["projects"], min(num_projects, len(config["projects"])))
    projects_text = "\n".join(f"- {p}" for p in projects)

    # Pick responsibilities (3-5 per role)
    resp1 = random.sample(config["responsibilities"], random.randint(3, 5))
    resp2 = random.sample(config["responsibilities"], random.randint(2, 4))
    resp1_text = "\n".join(f"- {r}" for r in resp1)
    resp2_text = "\n".join(f"- {r}" for r in resp2)

    # Pick certifications (1-3)
    num_certs = random.randint(1, 3)
    certs = random.sample(config["certifications"], min(num_certs, len(config["certifications"])))
    certs_text = "\n".join(f"- {c}" for c in certs)

    # Role titles
    title_variants = {
        "Data Scientist": ["Data Scientist", "Senior Data Scientist", "Lead Data Scientist", "Data Analyst", "Applied Scientist", "Quantitative Analyst"],
        "Software Engineer": ["Software Engineer", "Senior Software Engineer", "Staff Software Engineer", "Backend Engineer", "Platform Engineer", "Software Developer"],
        "Web Developer": ["Web Developer", "Frontend Developer", "Full Stack Developer", "Senior Web Developer", "UI Engineer", "Frontend Engineer"],
        "DevOps Engineer": ["DevOps Engineer", "Senior DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Infrastructure Engineer", "Platform Engineer"],
        "ML Engineer": ["ML Engineer", "Senior ML Engineer", "Machine Learning Engineer", "AI Engineer", "MLOps Engineer", "Applied ML Engineer"],
    }

    role_title = random.choice(title_variants[role_name])
    prev_titles = [t for t in title_variants[role_name] if t != role_title]
    prev_title = random.choice(prev_titles) if prev_titles else "Engineer"

    # Build resume from template
    template = random.choice(RESUME_TEMPLATES)
    linkedin = f"{first.lower()}{last.lower()}{random.randint(1,999)}"
    github = f"{first.lower()}{last.lower()}"

    resume = template.format(
        name=name,
        email=random_email(first, last),
        phone=random_phone(),
        linkedin=linkedin,
        github=github,
        summary=generate_summary(role_name, years_exp, skills_sample),
        degree=random.choice(DEGREES),
        university=random.choice(UNIVERSITIES),
        gpa=round(random.uniform(3.0, 4.0), 2),
        grad_year=random.randint(2010, 2025),
        role_title=role_title,
        company1=companies[0],
        date1=random_date_range(years_exp),
        responsibilities1=resp1_text,
        prev_title=prev_title,
        company2=companies[1],
        date2=random_prev_date(years_exp),
        responsibilities2=resp2_text,
        projects=projects_text,
        skills=", ".join(skills_sample),
        certifications=certs_text,
    )

    return resume


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("SYNTHETIC RESUME GENERATOR")
    print("=" * 60)

    # Clear existing generated resumes (keep originals)
    existing = list(OUTPUT_DIR.glob("resume_*.txt"))
    if existing:
        print(f"\nRemoving {len(existing)} existing resumes...")
        for f in existing:
            f.unlink()

    # Generate resumes
    all_entries = []
    total = 0

    for role_name, config in ROLE_CONFIG.items():
        role_id = config["id"]
        slug = role_name.lower().replace(" ", "_")

        print(f"\n[{role_name}] Generating {RESUMES_PER_CLASS} resumes...")

        for i in range(1, RESUMES_PER_CLASS + 1):
            filename = f"resume_{slug}_{i:03d}.txt"
            filepath = OUTPUT_DIR / filename

            resume_text = generate_resume(role_name, config, i)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(resume_text)

            all_entries.append({
                "filename": filename,
                "label": role_name,
                "label_id": role_id
            })
            total += 1

    # Write dataset.csv
    print(f"\n[CSV] Writing {len(all_entries)} entries to {CSV_PATH}")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "label_id"])
        writer.writeheader()
        writer.writerows(all_entries)

    # Verify
    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"   Total resumes: {total}")
    print(f"   Per class:     {RESUMES_PER_CLASS}")
    print(f"   Classes:       {len(ROLE_CONFIG)}")
    print(f"   CSV:           {CSV_PATH}")
    print(f"   Output dir:    {OUTPUT_DIR}")

    # Show per-class breakdown
    for role_name, config in ROLE_CONFIG.items():
        slug = role_name.lower().replace(" ", "_")
        count = len(list(OUTPUT_DIR.glob(f"resume_{slug}_*.txt")))
        print(f"   {role_name:<20}: {count} resumes")


if __name__ == "__main__":
    main()
