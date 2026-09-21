import os
import re
import yaml
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ConfigurableJobRanker:
    """
    Computes transparent multi-component compatibility scores based on configured matrix:
    Required Skills (40%), Responsibilities (25%), Experience (15%), Education (10%), Preferred Skills (10%).
    """

    DEFAULT_WEIGHTS = {
        "required_skills": 0.40,
        "responsibilities": 0.25,
        "experience_level": 0.15,
        "education_level": 0.10,
        "preferred_skills": 0.10
    }

    def __init__(self, config_path: str = "config/scoring.yaml"):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.load_config(config_path)

    def load_config(self, config_path: str):
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg and "scoring_weights" in cfg:
                        self.weights.update(cfg["scoring_weights"])
                        logger.info(f"Loaded scoring weights from {config_path}")
            except Exception as e:
                logger.warning(f"Could not parse {config_path}: {e}. Using default weights.")

        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            for k in self.weights:
                self.weights[k] /= total_w

    def compute_composite_score(
        self,
        semantic_responsibility_sim: float,
        required_skill_coverage: float,
        preferred_skill_coverage: float,
        resume_text: str,
        job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates individual component scores and final weighted composite compatibility score.
        """
        exp_score = self._evaluate_experience_alignment(resume_text, job.get("description", ""))
        edu_score = self._evaluate_education_alignment(resume_text, job.get("description", ""))

        w_req = self.weights.get("required_skills", 0.40) * required_skill_coverage
        w_resp = self.weights.get("responsibilities", 0.25) * semantic_responsibility_sim
        w_exp = self.weights.get("experience_level", 0.15) * exp_score
        w_edu = self.weights.get("education_level", 0.10) * edu_score
        w_pref = self.weights.get("preferred_skills", 0.10) * preferred_skill_coverage

        composite = (w_req + w_resp + w_exp + w_edu + w_pref)
        overall_percentage = min(100.0, max(0.0, composite * 100.0))

        return {
            "overall_match_score": round(overall_percentage, 1),
            "composite_score": round(composite, 4),
            "component_breakdown": {
                "required_skill_coverage": round(required_skill_coverage * 100, 1),
                "responsibility_semantic_similarity": round(semantic_responsibility_sim * 100, 1),
                "experience_alignment": round(exp_score * 100, 1),
                "education_alignment": round(edu_score * 100, 1),
                "preferred_skill_coverage": round(preferred_skill_coverage * 100, 1)
            },
            "configured_weights": self.weights
        }

    @staticmethod
    def _evaluate_experience_alignment(resume_text: str, jd_text: str) -> float:
        """
        Heuristic evaluation of years of experience alignment between candidate and JD.
        """
        jd_years = re.findall(r'(\d+)\+?\s*(?:years|yrs)', jd_text.lower())
        res_years = re.findall(r'(\d+)\+?\s*(?:years|yrs)', resume_text.lower())

        req_years = max([int(y) for y in jd_years], default=2)
        cand_years = max([int(y) for y in res_years], default=2)

        if cand_years >= req_years:
            return 1.0
        elif cand_years > 0:
            return cand_years / float(req_years)
        return 0.6

    @staticmethod
    def _evaluate_education_alignment(resume_text: str, jd_text: str) -> float:
        """
        Evaluates degree level alignment (PhD, Master's, Bachelor's).
        """
        res_low = resume_text.lower()
        jd_low = jd_text.lower()

        res_phd = "phd" in res_low or "doctorate" in res_low or "ph.d" in res_low
        res_ms = "master" in res_low or "m.s" in res_low or "m.tech" in res_low or "ms" in res_low
        res_bs = "bachelor" in res_low or "b.s" in res_low or "b.tech" in res_low or "bs" in res_low

        jd_phd = "phd" in jd_low or "doctorate" in jd_low
        jd_ms = "master" in jd_low or "m.s" in jd_low

        if jd_phd:
            if res_phd: return 1.0
            if res_ms: return 0.8
            if res_bs: return 0.5
            return 0.4
        elif jd_ms:
            if res_phd or res_ms: return 1.0
            if res_bs: return 0.85
            return 0.6

        return 1.0 if (res_bs or res_ms or res_phd) else 0.8
