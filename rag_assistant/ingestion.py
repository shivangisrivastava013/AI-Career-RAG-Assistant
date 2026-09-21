import json
import os
import glob
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class JobCorpusIngestor:
    """
    Ingests, validates, and normalizes job descriptions from JSON files or data directories.
    """

    @staticmethod
    def load_job_file(file_path: str) -> Dict[str, Any]:
        """
        Loads and validates a single job JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = ["job_id", "title", "company", "description"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Job JSON at {file_path} missing required key '{key}'")

        return {
            "job_id": str(data.get("job_id")),
            "title": str(data.get("title")),
            "company": str(data.get("company", "Unknown")),
            "location": str(data.get("location", "Remote / Unspecified")),
            "description": str(data.get("description")),
            "source_url": str(data.get("source_url", "")),
            "required_skills": list(data.get("required_skills", [])),
            "preferred_skills": list(data.get("preferred_skills", []))
        }

    @classmethod
    def load_corpus(cls, directory_path: str) -> List[Dict[str, Any]]:
        """
        Loads all job JSON files in a directory.
        """
        if not os.path.exists(directory_path):
            logger.warning(f"Job directory '{directory_path}' does not exist.")
            return []

        pattern = os.path.join(directory_path, "*.json")
        json_files = glob.glob(pattern)
        jobs = []

        for filepath in sorted(json_files):
            try:
                job = cls.load_job_file(filepath)
                jobs.append(job)
            except Exception as e:
                logger.error(f"Error loading job file {filepath}: {e}")

        logger.info(f"Loaded {len(jobs)} jobs from {directory_path}")
        return jobs
