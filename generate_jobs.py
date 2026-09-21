import json
import os

os.makedirs("data/jobs", exist_ok=True)

role_categories = [
    (
        "Machine Learning Engineer",
        ["Python", "PyTorch", "Scikit-Learn", "Docker", "PostgreSQL"],
        ["AWS", "MLflow", "CI/CD"],
    ),
    (
        "Applied AI Engineer",
        ["Python", "PyTorch", "Retrieval-Augmented Generation (RAG)", "FAISS", "FastAPI"],
        ["LangChain", "Docker", "Vector Databases"],
    ),
    ("Data Scientist", ["Python", "Scikit-Learn", "Pandas", "NumPy", "SQL"], ["PostgreSQL", "AWS", "Problem Solving"]),
    ("Data Engineer", ["Python", "SQL", "Spark / PySpark", "Airflow", "PostgreSQL"], ["Kafka", "AWS", "Docker"]),
    ("MLOps Engineer", ["Python", "Docker", "Kubernetes", "CI/CD", "MLflow"], ["AWS", "PostgreSQL", "Bash / Shell"]),
    (
        "Computer Vision Engineer",
        ["Python", "C++", "PyTorch", "OpenCV", "YOLO / Computer Vision"],
        ["CUDA", "Segment Anything (SAM)", "Docker"],
    ),
    (
        "NLP Engineer",
        ["Python", "PyTorch", "Transformers", "Large Language Models (LLMs)", "FastAPI"],
        ["HuggingFace", "RAG", "LangChain"],
    ),
    ("Robotics Engineer", ["Python", "C++", "ROS 2", "Gazebo", "Docker"], ["YOLO / Computer Vision", "Zenoh", "Linux"]),
    (
        "GenAI Engineer",
        ["Python", "Large Language Models (LLMs)", "Retrieval-Augmented Generation (RAG)", "FAISS", "LangChain"],
        ["Vector Databases", "Transformers", "Docker"],
    ),
    (
        "Data Analyst",
        ["SQL", "Python", "Pandas", "Problem Solving", "Communication"],
        ["PostgreSQL", "Bash / Shell", "Team Collaboration"],
    ),
]

companies = [
    "CognitiveTech",
    "ApexVision",
    "GraphLabs",
    "NovaRobotics",
    "SynthetixAI",
    "SpatialVerse",
    "DataStream",
    "EnterpriseAI",
    "ScaleML",
    "OmniData",
]

jobs = []
job_counter = 1

for cat_title, req_skills, pref_skills in role_categories:
    for comp in companies[:5]:
        job_id = f"job_{job_counter:03d}"
        jobs.append(
            {
                "job_id": job_id,
                "title": f"{cat_title} ({comp})",
                "company": f"{comp} Systems",
                "location": "New York, NY / Remote",
                "description": f"Responsibilities: Design, evaluate, and deploy scalable solutions for {cat_title} roles. Build production pipelines, optimize latency, and collaborate across engineering teams. Qualifications: Strong proficiency in {', '.join(req_skills[:3])}. Production experience with {', '.join(req_skills[3:])}. Preferred: Familiarity with {', '.join(pref_skills)}.",
                "source_url": None,
                "synthetic": True,
                "required_skills": req_skills,
                "preferred_skills": pref_skills,
            }
        )
        job_counter += 1

for job in jobs:
    filepath = os.path.join("data/jobs", f"{job['job_id']}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)

print(f"Successfully generated {len(jobs)} synthetic job records across 10 role categories in data/jobs/")
