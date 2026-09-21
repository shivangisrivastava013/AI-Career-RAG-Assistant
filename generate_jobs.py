import json
import os

jobs = [
    {
        "job_id": "job_001",
        "title": "Machine Learning Engineer - RAG & Applied AI",
        "company": "Cognitive AI Systems",
        "location": "New York, NY (Hybrid)",
        "description": "Responsibilities: Design, build, and deploy production-grade Retrieval-Augmented Generation (RAG) pipelines using Sentence Transformers, FAISS, and LangChain. Optimize vector database indexing and search latency for large document corpora. Qualifications: 2+ years of production PyTorch experience. Strong knowledge of PostgreSQL, Vector Databases, Docker, and REST APIs. Preferred: Experience with ROS 2 and Gazebo simulation.",
        "source_url": "https://careers.cognitiveai.com/jobs/001",
        "required_skills": ["Python", "PyTorch", "RAG", "FAISS", "Vector Databases", "Docker", "PostgreSQL"],
        "preferred_skills": ["LangChain", "ROS 2", "AWS"]
    },
    {
        "job_id": "job_002",
        "title": "Computer Vision & Visual Perception Engineer",
        "company": "Apex Vision Robotics",
        "location": "San Francisco, CA",
        "description": "Responsibilities: Develop real-time object detection and visual segmentation pipelines combining YOLOv8 and Segment Anything (SAM 2). Implement zero-shot visual perception and CUDA acceleration on edge devices. Qualifications: Strong Python and C++ experience. PyTorch, OpenCV, Docker, and CUDA experience. Preferred: ROS 2 integration experience.",
        "source_url": "https://careers.apexvision.com/jobs/002",
        "required_skills": ["Python", "C++", "PyTorch", "OpenCV", "CUDA", "YOLO / Computer Vision", "Docker"],
        "preferred_skills": ["Segment Anything (SAM)", "ROS 2", "TensorRT"]
    },
    {
        "job_id": "job_003",
        "title": "Graph Machine Learning Research Engineer",
        "company": "GraphNet Labs",
        "location": "Boston, MA",
        "description": "Responsibilities: Benchmark and scale Graph Neural Network (GCN, GraphSAGE) architectures for node classification and citation networks. Implement graph embeddings using PyTorch Geometric. Qualifications: MS or PhD in Computer Science or AI. PyTorch, PyTorch Geometric, Python, SQL. Preferred: Scikit-Learn and Docker.",
        "source_url": "https://careers.graphnet.com/jobs/003",
        "required_skills": ["Python", "PyTorch", "Graph Neural Networks", "PyTorch Geometric", "SQL"],
        "preferred_skills": ["Scikit-Learn", "Docker", "Bash / Shell"]
    },
    {
        "job_id": "job_004",
        "title": "Autonomous Robotics Perception & Navigation Engineer",
        "company": "Nova Robotics Inc.",
        "location": "Newark, NJ",
        "description": "Responsibilities: Build autonomous TurtleBot perception and trajectory navigation packages in ROS 2. Integrate Zenoh telemetry streaming and real-time Gazebo simulations. Qualifications: ROS 2, Python, C++, Gazebo, Docker, Linux. Preferred: 3D Gaussian Splatting and COLMAP.",
        "source_url": "https://careers.novarobotics.com/jobs/004",
        "required_skills": ["Python", "C++", "ROS 2", "Gazebo", "Docker", "Linux"],
        "preferred_skills": ["Zenoh", "3D Gaussian Splatting", "PostgreSQL"]
    },
    {
        "job_id": "job_005",
        "title": "NLP & Generative AI Engineer",
        "company": "Synthetix AI",
        "location": "Remote",
        "description": "Responsibilities: Build abstractive and extractive text summarization engines using HuggingFace Transformers and LLMs. Implement fine-tuning pipelines for domain-specific text generation. Qualifications: Python, PyTorch, Transformers, HuggingFace, FastAPI, REST API. Preferred: LangChain, Vector Databases.",
        "source_url": "https://careers.synthetix.com/jobs/005",
        "required_skills": ["Python", "PyTorch", "Transformers", "Large Language Models (LLMs)", "FastAPI"],
        "preferred_skills": ["LangChain", "RAG", "Docker"]
    },
    {
        "job_id": "job_006",
        "title": "3D Neural Rendering & Spatial AI Engineer",
        "company": "SpatialVerse Systems",
        "location": "Austin, TX",
        "description": "Responsibilities: Implement 3D Gaussian Splatting and neural rendering pipelines for indoor environment reconstruction. Translate Polycam captures into COLMAP camera poses. Qualifications: Python, PyTorch, COLMAP, 3D Gaussian Splatting, OpenCV, CUDA. Preferred: ROS 2 integration.",
        "source_url": "https://careers.spatialverse.com/jobs/006",
        "required_skills": ["Python", "PyTorch", "CUDA", "OpenCV", "3D Gaussian Splatting", "COLMAP"],
        "preferred_skills": ["ROS 2", "Docker"]
    },
    {
        "job_id": "job_007",
        "title": "MLOps & Data Platform Engineer",
        "company": "Datastream AI",
        "location": "Seattle, WA",
        "description": "Responsibilities: Build automated CI/CD and MLOps deployment pipelines for deep learning models. Manage PostgreSQL databases, Redis caches, and Docker/Kubernetes clusters. Qualifications: Python, Docker, Kubernetes, CI/CD, PostgreSQL, Redis, AWS. Preferred: PySpark, MLflow.",
        "source_url": "https://careers.datastream.com/jobs/007",
        "required_skills": ["Python", "Docker", "Kubernetes", "PostgreSQL", "CI/CD", "AWS"],
        "preferred_skills": ["Redis", "PySpark", "MLflow"]
    },
    {
        "job_id": "job_008",
        "title": "Senior AI Systems Architect",
        "company": "Enterprise AI Global",
        "location": "New York, NY",
        "description": "Responsibilities: Architect end-to-end enterprise AI products leveraging LLMs, RAG, vector search, and scalable microservices. Qualifications: 5+ years experience in Python, System Design, RAG, PyTorch, AWS, PostgreSQL, Docker. Preferred: Kubernetes.",
        "source_url": "https://careers.enterpriseai.com/jobs/008",
        "required_skills": ["Python", "System Design", "RAG", "PyTorch", "AWS", "PostgreSQL"],
        "preferred_skills": ["Kubernetes", "FAISS"]
    }
]

# Generate additional job JSON files dynamically up to 30 jobs
roles = [
    ("ML Engineer - LLM Infrastructure", "DataCore Systems", ["Python", "PyTorch", "Transformers", "Docker", "AWS"]),
    ("Robotics Motion Control Software Engineer", "Kinematics Tech", ["C++", "Python", "ROS 2", "Gazebo", "Control Systems"]),
    ("AI Research Scientist - Computer Vision", "Visionary Research", ["PyTorch", "OpenCV", "CUDA", "Python", "Deep Learning"]),
    ("Senior Applied NLP Engineer", "TextMetrics Inc", ["Python", "Transformers", "NLP", "FastAPI", "PostgreSQL"]),
    ("Generative AI Solutions Architect", "CloudAI Partners", ["Python", "RAG", "LangChain", "Vector Databases", "AWS"]),
    ("Data Engineer - AI Pipelines", "OmniData Solutions", ["Python", "SQL", "Spark / PySpark", "Airflow", "PostgreSQL"]),
    ("Full Stack AI Software Engineer", "SmartApps Corp", ["Python", "TypeScript", "React", "FastAPI", "Docker"]),
    ("Computer Vision Algorithm Developer", "EdgeVision Systems", ["C++", "Python", "OpenCV", "PyTorch", "YOLO / Computer Vision"]),
    ("Autonomous Navigation Systems Engineer", "DriveTech AI", ["ROS 2", "C++", "Python", "Gazebo", "LiDAR"]),
    ("Machine Learning Platform Developer", "ScaleML Labs", ["Python", "Docker", "Kubernetes", "MLflow", "CI/CD"]),
    ("Applied Scientist - Graph Neural Networks", "NetworkAI", ["PyTorch", "PyTorch Geometric", "Graph Neural Networks", "Python"]),
    ("AI Security & Watermarking Researcher", "CyberGuard AI", ["Python", "PyTorch", "TensorFlow", "Cybersecurity", "Autoencoders"]),
    ("Robotic Perception Software Specialist", "BotSense Inc", ["ROS 2", "Python", "YOLO / Computer Vision", "Gazebo", "Docker"]),
    ("Lead AI Search & Information Retrieval Engineer", "FindIT AI", ["RAG", "FAISS", "Python", "Elasticsearch", "Sentence Transformers"]),
    ("ML Optimization & Quantization Engineer", "TensorCore", ["PyTorch", "CUDA", "TensorRT", "C++", "Python"]),
    ("Deep Learning Engineer - Audio & Speech", "Soundwave AI", ["Python", "PyTorch", "Transformers", "Audio Processing"]),
    ("Robotics Simulation Engineer", "SimuRobot Technologies", ["ROS 2", "Gazebo", "C++", "Python", "Docker"]),
    ("Generative Multimodal AI Engineer", "OmniVision AI", ["PyTorch", "Transformers", "Segment Anything (SAM)", "Diffusion Models"]),
    ("Junior AI Developer", "LaunchPad AI", ["Python", "Scikit-Learn", "Git / GitHub", "SQL", "REST API"]),
    ("Staff AI Engineer - Enterprise Search", "Apex Search", ["Python", "RAG", "FAISS", "Vector Databases", "PostgreSQL", "Docker"]),
    ("Computer Vision Software Engineer", "CamTech Solutions", ["Python", "OpenCV", "PyTorch", "C++"]),
    ("MLOps Infrastructure Specialist", "CloudDeploy AI", ["Docker", "Kubernetes", "CI/CD", "Python", "Terraform"])
]

for idx, (title, comp, skills) in enumerate(roles, start=9):
    job_id = f"job_{idx:03d}"
    jobs.append({
        "job_id": job_id,
        "title": title,
        "company": comp,
        "location": "New York, NY / Remote",
        "description": f"Responsibilities: Design, evaluate, and scale production systems for {title}. Build robust pipelines and collaborate with cross-functional AI teams. Qualifications: Proficiency in {', '.join(skills[:3])}. Experience with {', '.join(skills[3:])}.",
        "source_url": f"https://careers.example.com/jobs/{job_id}",
        "required_skills": skills[:3],
        "preferred_skills": skills[3:]
    })

os.makedirs("data/jobs", exist_ok=True)
for job in jobs:
    filepath = os.path.join("data/jobs", f"{job['job_id']}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)

print(f"Successfully generated {len(jobs)} job JSON records in data/jobs/")
