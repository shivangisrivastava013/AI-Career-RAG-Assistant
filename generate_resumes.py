import os

os.makedirs("data/resumes", exist_ok=True)

resume_text = """Shivangi Srivastava
AI & Machine Learning Engineer | MS in Artificial Intelligence at NJIT
Email: goforshivangi@gmail.com | Phone: +1 (848) 315-8969
LinkedIn: linkedin.com/in/shivangisrivastava013 | Location: Newark, NJ

PROFESSIONAL SUMMARY
Artificial Intelligence Engineer and MS candidate at NJIT (3.8/4.0 GPA) with a B.Tech in CSE (AI & ML) from Manipal University. Specialized in production-grade Retrieval-Augmented Generation (RAG), Graph Neural Networks (GCN/GraphSAGE), zero-shot computer vision (YOLOv8, SAM 2), and ROS 2 autonomous robotics perception pipelines with empirical rigor.

EDUCATION
- Master of Science in Artificial Intelligence | New Jersey Institute of Technology (NJIT)
  GPA: 3.8 / 4.0 | Sep 2025 - Dec 2026 | President, GWiCS
  Coursework: Machine Learning, Deep Learning, NLP, Graph Neural Networks, AI for Robotics, Federated Machine Learning.
- Bachelor of Technology (Hons.) in CSE (AI & ML) | Manipal University, Jaipur
  GPA: 8.14 / 10 | Aug 2021 - Jul 2025

TECHNICAL SKILLS
- Programming & Frameworks: Python, C++, PyTorch, TensorFlow, Scikit-Learn, OpenCV, PyTorch Geometric, CUDA.
- Generative AI & NLP: RAG, Sentence Transformers, FAISS, Vector Databases, Transformers, LangChain, LLMs.
- Robotics & Perception: ROS 2, Gazebo, YOLO, Segment Anything (SAM 2), 3D Gaussian Splatting, COLMAP, Zenoh.
- Cloud & Infrastructure: Docker, PostgreSQL, SQL, Git/GitHub, Linux, REST API, FastAPI.

KEY PROJECTS
- ShadowTag (Deep Learning Image Watermarking): Trained convolutional autoencoder; achieved 34.7 dB PSNR, 0.96 SSIM, and 94.2% watermark recovery (92% after JPEG Q=70 compression).
- CareerLens (AI Career RAG Assistant): Built RAG vector retrieval system over 100 job descriptions reaching 0.88 Recall@5 and 0.81 MRR.
- Autonomous Robotic Perception Pipeline: Built TurtleBot perception in Gazebo using ROS 2, YOLOv8, Zenoh, and PostgreSQL logging (0.91 mAP@50 at 18 FPS).
- Real-Time Multi-Stage Vision Pipeline: Combined YOLO and SAM 2 across 1,500 annotated images (0.89 mAP@50, 0.87 mean IoU at 14 FPS).
- Graph Neural Network Benchmarking: GraphSAGE node classification reaching 88.1% accuracy and 0.87 F1 score.
- 3D Gaussian Splatting Navigation: Reconstructed 3D scenes from 150 Polycam images (29.8 dB PSNR, 0.94 SSIM).
"""

# Save TXT
with open("data/resumes/sample_resume.txt", "w", encoding="utf-8") as f:
    f.write(resume_text)
print("Saved data/resumes/sample_resume.txt")

# Save DOCX if python-docx installed
try:
    import docx
    doc = docx.Document()
    for line in resume_text.split("\n"):
        if line.strip():
            doc.add_paragraph(line)
    doc.save("data/resumes/sample_resume.docx")
    print("Saved data/resumes/sample_resume.docx")
except Exception as e:
    print(f"DOCX generation deferred: {e}")

# Save PDF if ReportLab installed
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas("data/resumes/sample_resume.pdf", pagesize=letter)
    y = 750
    for line in resume_text.split("\n"):
        if y < 40:
            c.showPage()
            y = 750
        c.drawString(40, y, line[:90])
        y -= 14
    c.save()
    print("Saved data/resumes/sample_resume.pdf")
except Exception as e:
    print(f"PDF generation deferred: {e}")
