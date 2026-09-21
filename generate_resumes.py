import os

os.makedirs("data/resumes", exist_ok=True)

candidate_resumes = {
    "sample_resume.txt": """Shivangi Srivastava
AI & Machine Learning Engineer | MS in Artificial Intelligence at NJIT
Specialized in Retrieval-Augmented Generation (RAG), Graph Neural Networks (GCN/GraphSAGE), zero-shot computer vision (YOLOv8, SAM 2), and ROS 2 autonomous robotics perception.
SKILLS: Python, C++, PyTorch, TensorFlow, Scikit-Learn, OpenCV, PyTorch Geometric, CUDA, Retrieval-Augmented Generation (RAG), Sentence Transformers, FAISS, Vector Databases, Transformers, LangChain, Large Language Models (LLMs), ROS 2, Gazebo, YOLO / Computer Vision, Segment Anything (SAM), 3D Gaussian Splatting, COLMAP, Docker, PostgreSQL, SQL, Git / GitHub, Linux, FastAPI.""",
    "resume_ml_engineer.txt": """Alex Rivera
Machine Learning Engineer
Experienced ML Engineer specializing in production PyTorch model deployment, Scikit-Learn pipelines, and PostgreSQL data pipelines.
SKILLS: Python, PyTorch, Scikit-Learn, Docker, PostgreSQL, AWS, MLflow, CI/CD, SQL, Git / GitHub.""",
    "resume_ai_engineer.txt": """Elena Rostova
Applied AI Engineer
Specialized in building enterprise Retrieval-Augmented Generation (RAG) systems, vector search with FAISS, and LLM orchestration with LangChain and FastAPI.
SKILLS: Python, PyTorch, Retrieval-Augmented Generation (RAG), FAISS, FastAPI, LangChain, Docker, Vector Databases, Large Language Models (LLMs).""",
    "resume_data_scientist.txt": """David Chen
Data Scientist
Statistical modeling expert proficient in Python, Pandas, NumPy, Scikit-Learn, and SQL analytics.
SKILLS: Python, Scikit-Learn, Pandas, NumPy, SQL, PostgreSQL, AWS, Problem Solving, Communication.""",
    "resume_data_engineer.txt": """Marcus Johnson
Data Engineer
Data infrastructure engineer building distributed ETL pipelines with Spark, Airflow, and PostgreSQL.
SKILLS: Python, SQL, Spark / PySpark, Airflow, PostgreSQL, Kafka, AWS, Docker.""",
    "resume_mlops_engineer.txt": """Sophia Taylor
MLOps Engineer
DevOps and MLOps specialist managing Docker containers, Kubernetes deployment, and automated CI/CD pipelines.
SKILLS: Python, Docker, Kubernetes, CI/CD, MLflow, AWS, PostgreSQL, Bash / Shell, Git / GitHub.""",
    "resume_cv_engineer.txt": """Hiroshi Tanaka
Computer Vision Engineer
Vision researcher building real-time object detection models using PyTorch, OpenCV, YOLO, and SAM.
SKILLS: Python, C++, PyTorch, OpenCV, YOLO / Computer Vision, CUDA, Segment Anything (SAM), Docker.""",
    "resume_nlp_engineer.txt": """Amina Vance
NLP Engineer
Natural Language Processing engineer building Transformer models, text summarization, and HuggingFace pipelines.
SKILLS: Python, PyTorch, Transformers, Large Language Models (LLMs), FastAPI, HuggingFace, RAG, LangChain.""",
    "resume_robotics_engineer.txt": """Liam O'Connor
Robotics Autonomous Navigation Engineer
Robotics engineer developing ROS 2 TurtleBot navigation, Gazebo simulations, and C++ perception modules.
SKILLS: Python, C++, ROS 2, Gazebo, Docker, YOLO / Computer Vision, Zenoh, Linux.""",
    "resume_genai_engineer.txt": """Priya Sharma
Generative AI Engineer
Generative AI specialist building LLM applications, vector search architectures, and RAG pipelines.
SKILLS: Python, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), FAISS, LangChain, Vector Databases, Transformers, Docker.""",
}

for filename, content in candidate_resumes.items():
    filepath = os.path.join("data/resumes", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

# Also generate DOCX and PDF for sample_resume
try:
    import docx

    doc = docx.Document()
    for line in candidate_resumes["sample_resume.txt"].split("\n"):
        if line.strip():
            doc.add_paragraph(line)
    doc.save("data/resumes/sample_resume.docx")
except Exception:
    pass

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas("data/resumes/sample_resume.pdf", pagesize=letter)
    y = 750
    for line in candidate_resumes["sample_resume.txt"].split("\n"):
        if y < 40:
            c.showPage()
            y = 750
        c.drawString(40, y, line[:90])
        y -= 14
    c.save()
except Exception:
    pass

print(f"Successfully generated {len(candidate_resumes)} candidate resumes in data/resumes/")
