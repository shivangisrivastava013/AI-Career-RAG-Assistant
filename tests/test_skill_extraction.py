import pytest
from rag_assistant.skill_extraction import CategorizedSkillExtractor


def test_categorized_skill_extraction():
    text = "Developed image classification pipelines using PyTorch and OpenCV. Deployed RAG applications with FAISS and PostgreSQL on AWS."
    skills = CategorizedSkillExtractor.extract_categorized_skills(text)

    assert "PyTorch" in skills
    assert skills["PyTorch"]["category"] == "ml_frameworks"
    assert "OpenCV" in skills
    assert "Retrieval-Augmented Generation (RAG)" in skills
    assert skills["Retrieval-Augmented Generation (RAG)"]["category"] == "genai_technologies"
    assert "PostgreSQL" in skills
    assert "AWS" in skills


def test_skill_alias_normalization():
    norm_llm = CategorizedSkillExtractor.normalize_skill_name("llms")
    assert norm_llm == "Large Language Models (LLMs)"

    norm_torch = CategorizedSkillExtractor.normalize_skill_name("torch")
    assert norm_torch == "PyTorch"

    norm_postgres = CategorizedSkillExtractor.normalize_skill_name("postgres")
    assert norm_postgres == "PostgreSQL"


def test_skill_comparison():
    resume_text = "Proficient in Python, PyTorch, Docker, PostgreSQL."
    jd_req = ["Python", "PyTorch", "ROS 2", "CUDA"]
    jd_text = "Looking for a engineer with Python, PyTorch, ROS 2, and CUDA experience."

    comp = CategorizedSkillExtractor.compare_skills(
        resume_text=resume_text,
        job_required_skills=jd_req,
        job_preferred_skills=["Docker"],
        job_description_text=jd_text
    )

    matching_names = [m["skill"] for m in comp["matching_skills"]]
    missing_names = [g["skill"] for g in comp["missing_required_skills"]]

    assert "Python" in matching_names or "PyTorch" in matching_names
    assert "ROS 2" in missing_names or "CUDA" in missing_names
    assert comp["required_coverage"] > 0.0
