import re
from typing import Dict, List, Any
from .embeddings import VectorStoreManager
from .parser import DocumentParser


class CareerRAGMatcher:
    """
    RAG-powered candidate evaluation engine for matching resumes against
    job descriptions and identifying skill gaps.
    """

    CORE_SKILLS_TAXONOMY = [
        "python", "pytorch", "tensorflow", "rag", "langchain", "faiss", "llm", "llms",
        "transformers", "nlp", "computer vision", "gnn", "graph neural networks",
        "ros2", "docker", "cuda", "sql", "postgresql", "fastapi", "rest api",
        "machine learning", "deep learning", "c++", "git", "wsl2", "scikit-learn"
    ]

    def __init__(self, vector_store: VectorStoreManager = None):
        self.vector_store = vector_store or VectorStoreManager()

    def extract_skills_set(self, text: str) -> set:
        """
        Extracts recognized technical skills from text string based on taxonomy.
        """
        text_lower = text.lower()
        found_skills = set()
        for skill in self.CORE_SKILLS_TAXONOMY:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        return found_skills

    def evaluate_match(self, resume_text: str, job_description_text: str) -> Dict[str, Any]:
        """
        Evaluates semantic similarity, skill coverage, and gap analysis between resume and job description.
        """
        resume_clean = DocumentParser.clean_text(resume_text)
        jd_clean = DocumentParser.clean_text(job_description_text)

        # 1. Semantic Embedding Similarity Score
        resume_vec = self.vector_store.encode([resume_clean])[0]
        jd_vec = self.vector_store.encode([jd_clean])[0]
        semantic_score = VectorStoreManager.cosine_similarity(resume_vec, jd_vec)

        # 2. Skill Gap Extraction
        resume_skills = self.extract_skills_set(resume_clean)
        jd_skills = self.extract_skills_set(jd_clean)

        matching_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills.difference(resume_skills)

        skill_coverage = len(matching_skills) / max(1, len(jd_skills))

        # 3. Overall Composite Match Score (Weighted: 60% Semantic + 40% Skill Coverage)
        composite_score = (semantic_score * 0.60) + (skill_coverage * 0.40)
        overall_match_percentage = min(100.0, max(0.0, composite_score * 100.0))

        # 4. RAG Top Context Retrieval
        jd_chunks = DocumentParser.chunk_text(jd_clean, chunk_size=128, chunk_overlap=16)
        top_jd_matches = self.vector_store.retrieve_top_k("Key Requirements Skills Experience", jd_chunks, k=3)

        # 5. Career Recommendation
        recommendation = self._generate_recommendation(overall_match_percentage, list(missing_skills))

        return {
            "overall_match_percentage": round(overall_match_percentage, 1),
            "semantic_similarity_score": round(semantic_score, 4),
            "skill_coverage_ratio": round(skill_coverage, 4),
            "matching_skills": sorted(list(matching_skills)),
            "missing_skills": sorted(list(missing_skills)),
            "resume_skills_detected": sorted(list(resume_skills)),
            "jd_skills_detected": sorted(list(jd_skills)),
            "top_relevant_jd_context": [chunk for chunk, score in top_jd_matches],
            "recommendation": recommendation
        }

    def _generate_recommendation(self, match_pct: float, missing: List[str]) -> str:
        """
        Generates actionable feedback based on match score and missing requirements.
        """
        if match_pct >= 85.0:
            status = "[EXCELLENT MATCH] Strong alignment across core technical stack and experience."
        elif match_pct >= 70.0:
            status = "[GOOD MATCH] Solid foundation with minor skill gaps to address."
        elif match_pct >= 50.0:
            status = "[MODERATE MATCH] Meets basic prerequisites, but missing key specialized skills."
        else:
            status = "[LOW MATCH] Significant gaps in core required competencies."

        if missing:
            gap_str = f" To maximize match score, consider adding experience or projects in: {', '.join(missing[:5])}."
        else:
            gap_str = " Candidate covers all detected key job description requirements!"

        return status + gap_str
