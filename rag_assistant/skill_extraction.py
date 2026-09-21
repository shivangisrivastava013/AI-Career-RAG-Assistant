import re
from typing import Dict, List, Set, Any, Tuple


class CategorizedSkillExtractor:
    """
    Extracts normalized technical and soft skills across 9 categories with canonical
    alias resolution and surrounding sentence evidence extraction.
    """

    SKILL_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
        "programming_languages": {
            "Python": ["python", "py"],
            "C++": ["c++", "cpp"],
            "Java": ["java"],
            "TypeScript": ["typescript", "ts"],
            "JavaScript": ["javascript", "js"],
            "SQL": ["sql"],
            "Bash / Shell": ["bash", "shell", "sh"]
        },
        "ml_frameworks": {
            "PyTorch": ["pytorch", "torch"],
            "TensorFlow": ["tensorflow", "tf"],
            "Scikit-Learn": ["scikit-learn", "sklearn"],
            "OpenCV": ["opencv", "cv2"],
            "Keras": ["keras"],
            "JAX": ["jax"]
        },
        "genai_technologies": {
            "Retrieval-Augmented Generation (RAG)": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
            "Large Language Models (LLMs)": ["llm", "llms", "large language model", "large language models"],
            "Transformers": ["transformer", "transformers", "huggingface", "bert", "gpt"],
            "FAISS": ["faiss"],
            "LangChain": ["langchain"],
            "LlamaIndex": ["llamaindex", "llama-index"],
            "Vector Databases": ["vector database", "vector DB", "chromadb", "pinecone", "qdrant", "weaviate"]
        },
        "databases": {
            "PostgreSQL": ["postgresql", "postgres"],
            "MongoDB": ["mongodb", "mongo"],
            "Redis": ["redis"],
            "MySQL": ["mysql"],
            "SQLite": ["sqlite"]
        },
        "cloud_platforms": {
            "AWS": ["aws", "amazon web services", "s3", "ec2"],
            "Google Cloud Platform (GCP)": ["gcp", "google cloud", "bigquery"],
            "Azure": ["azure", "microsoft azure"]
        },
        "devops_mlops": {
            "Docker": ["docker", "containerization"],
            "Kubernetes": ["kubernetes", "k8s"],
            "Git / GitHub": ["git", "github", "gitlab"],
            "MLflow": ["mlflow"],
            "CI/CD": ["ci/cd", "github actions", "jenkins"]
        },
        "data_engineering": {
            "Spark / PySpark": ["spark", "pyspark"],
            "Airflow": ["airflow"],
            "Kafka": ["kafka"],
            "Pandas": ["pandas"],
            "NumPy": ["numpy"]
        },
        "robotics": {
            "ROS 2": ["ros2", "ros 2", "robot operating system 2"],
            "ROS": ["ros", "robot operating system"],
            "Gazebo": ["gazebo"],
            "YOLO / Computer Vision": ["yolo", "yolov8", "object detection", "computer vision"],
            "Segment Anything (SAM)": ["sam", "sam2", "sam 2", "segment anything"]
        },
        "soft_skills": {
            "Problem Solving": ["problem solving", "analytical skills"],
            "Team Collaboration": ["collaboration", "teamwork", "cross-functional"],
            "Communication": ["communication", "technical writing", "presentation"]
        }
    }

    @classmethod
    def extract_categorized_skills(cls, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extracts matched skills categorized by domain, resolving aliases to canonical names.
        Returns Dict mapping canonical skill -> { category, aliases_matched, evidence }
        """
        if not text:
            return {}

        text_lower = text.lower()
        extracted: Dict[str, Dict[str, Any]] = {}

        for category, skill_map in cls.SKILL_TAXONOMY.items():
            for canonical_name, aliases in skill_map.items():
                for alias in aliases:
                    pattern = r'\b' + re.escape(alias) + r'\b'
                    match = re.search(pattern, text_lower)
                    if match:
                        evidence = cls._extract_sentence_context(text, match.start(), match.end())
                        if canonical_name not in extracted:
                            extracted[canonical_name] = {
                                "skill": canonical_name,
                                "category": category,
                                "alias_matched": alias,
                                "evidence": evidence
                            }
                        break

        return extracted

    @staticmethod
    def _extract_sentence_context(text: str, start_pos: int, end_pos: int) -> str:
        """
        Extracts surrounding sentence / context window around a keyword match.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cumulative_len = 0
        for s in sentences:
            s_len = len(s) + 1
            if cumulative_len <= start_pos <= cumulative_len + s_len:
                return s.strip()
            cumulative_len += s_len

        # Fallback snippet
        left = max(0, start_pos - 60)
        right = min(len(text), end_pos + 60)
        return text[left:right].strip()

    @classmethod
    def compare_skills(
        cls,
        resume_text: str,
        job_required_skills: List[str],
        job_preferred_skills: List[str],
        job_description_text: str = ""
    ) -> Dict[str, Any]:
        """
        Compares candidate resume skills against job required and preferred skill sets.
        Returns matching skills, missing required skills, missing preferred skills, and evidence.
        """
        resume_skills_dict = cls.extract_categorized_skills(resume_text)
        jd_skills_dict = cls.extract_categorized_skills(job_description_text)

        # Merge explicit Jd lists with extracted JD skills
        all_jd_required = set()
        for req in job_required_skills:
            all_jd_required.add(cls.normalize_skill_name(req))
        for canonical, info in jd_skills_dict.items():
            all_jd_required.add(canonical)

        all_jd_preferred = set()
        for pref in job_preferred_skills:
            all_jd_preferred.add(cls.normalize_skill_name(pref))

        resume_canonical_set = set(resume_skills_dict.keys())

        matching = []
        missing_required = []
        missing_preferred = []

        for skill in all_jd_required:
            if skill in resume_canonical_set:
                matching.append({
                    "skill": skill,
                    "category": resume_skills_dict[skill]["category"],
                    "resume_evidence": resume_skills_dict[skill]["evidence"],
                    "job_evidence": jd_skills_dict.get(skill, {}).get("evidence", "Required by job description.")
                })
            else:
                missing_required.append({
                    "skill": skill,
                    "category": jd_skills_dict.get(skill, {}).get("category", "general"),
                    "job_evidence": jd_skills_dict.get(skill, {}).get("evidence", "Required skill for role.")
                })

        for skill in all_jd_preferred:
            if skill not in resume_canonical_set and skill not in [m["skill"] for m in missing_required]:
                missing_preferred.append({
                    "skill": skill,
                    "category": jd_skills_dict.get(skill, {}).get("category", "general"),
                    "job_evidence": "Preferred skill for role."
                })

        req_coverage = len(matching) / max(1, len(all_jd_required))
        pref_coverage = (
            len([s for s in all_jd_preferred if s in resume_canonical_set]) / max(1, len(all_jd_preferred))
            if all_jd_preferred else 1.0
        )

        return {
            "matching_skills": matching,
            "missing_required_skills": missing_required,
            "missing_preferred_skills": missing_preferred,
            "required_coverage": req_coverage,
            "preferred_coverage": pref_coverage,
            "candidate_skills_by_category": cls._group_by_category(resume_skills_dict)
        }

    @classmethod
    def normalize_skill_name(cls, skill_str: str) -> str:
        """
        Maps raw skill string to canonical taxonomy name if alias exists.
        """
        s_lower = skill_str.lower().strip()
        for category, skill_map in cls.SKILL_TAXONOMY.items():
            for canonical, aliases in skill_map.items():
                if s_lower == canonical.lower() or s_lower in aliases:
                    return canonical
        return skill_str.strip().title()

    @staticmethod
    def _group_by_category(skills_dict: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}
        for canonical, info in skills_dict.items():
            cat = info["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(canonical)
        return grouped
