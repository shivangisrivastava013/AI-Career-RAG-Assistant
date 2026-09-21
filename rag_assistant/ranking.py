import logging
import os
import re
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigurableJobRanker:
    """
    Computes transparent multi-component compatibility scores using a 5-component matrix:
    Required Skills (40%), Responsibilities (25%), Experience (15%), Education (10%), Preferred Skills (10%).
    Includes dynamic weight redistribution for missing/unspecified components.
    """

    DEFAULT_WEIGHTS = {
        "required_skills": 0.40,
        "responsibilities": 0.25,
        "experience_level": 0.15,
        "education_level": 0.10,
        "preferred_skills": 0.10,
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

    def compute_composite_score(
        self,
        semantic_responsibility_sim: float,
        required_skill_coverage: float,
        preferred_skill_coverage: Optional[float],
        resume_text: str,
        job: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculates individual component scores and final weighted composite compatibility score
        with dynamic weight redistribution for unavailable components.
        """
        exp_score = self._evaluate_experience_alignment(resume_text, job.get("description", ""))
        edu_score = self._evaluate_education_alignment(resume_text, job.get("description", ""))

        raw_scores: Dict[str, Optional[float]] = {
            "required_skills": required_skill_coverage,
            "responsibilities": semantic_responsibility_sim,
            "experience_level": exp_score,
            "education_level": edu_score,
            "preferred_skills": preferred_skill_coverage,
        }

        # Filter active components (where score is not None)
        active_weights = {}
        for comp, score in raw_scores.items():
            if score is not None:
                active_weights[comp] = self.weights.get(comp, 0.10)

        # Dynamic weight redistribution
        total_active_w = sum(active_weights.values())
        norm_weights = {}
        if total_active_w > 0:
            for comp in active_weights:
                norm_weights[comp] = active_weights[comp] / total_active_w

        composite = 0.0
        component_breakdown = {}

        for comp, score in raw_scores.items():
            if score is not None:
                w = norm_weights.get(comp, 0.0)
                composite += w * score
                component_breakdown[comp] = round(score * 100, 1)
            else:
                component_breakdown[comp] = "insufficient_evidence"

        overall_percentage = min(100.0, max(0.0, composite * 100.0))

        return {
            "overall_match_score": round(overall_percentage, 1),
            "composite_score": round(composite, 4),
            "component_breakdown": component_breakdown,
            "redistributed_weights": {k: round(v, 4) for k, v in norm_weights.items()},
        }

    @staticmethod
    def _evaluate_experience_alignment(resume_text: str, jd_text: str) -> Optional[float]:
        """
        Evaluates years of experience alignment. Returns None if unspecified.
        """
        jd_years = re.findall(r"(\d+)\+?\s*(?:years|yrs)", jd_text.lower())
        res_years = re.findall(r"(\d+)\+?\s*(?:years|yrs)", resume_text.lower())

        if not jd_years or not res_years:
            return None

        req_years = max([int(y) for y in jd_years])
        cand_years = max([int(y) for y in res_years])

        if cand_years >= req_years:
            return 1.0
        elif req_years > 0:
            return cand_years / float(req_years)
        return 1.0

    @staticmethod
    def _evaluate_education_alignment(resume_text: str, jd_text: str) -> Optional[float]:
        """
        Evaluates degree level alignment using word-boundary regex patterns.
        """
        res_low = resume_text.lower()
        jd_low = jd_text.lower()

        phd_pattern = r"\b(?:ph\.?d\.?|doctorate)\b"
        ms_pattern = r"\b(?:m\.?s\.?|master(?:'s)?|m\.tech)\b"
        bs_pattern = r"\b(?:b\.?s\.?|bachelor(?:'s)?|b\.tech)\b"

        res_phd = bool(re.search(phd_pattern, res_low))
        res_ms = bool(re.search(ms_pattern, res_low))
        res_bs = bool(re.search(bs_pattern, res_low))

        jd_phd = bool(re.search(phd_pattern, jd_low))
        jd_ms = bool(re.search(ms_pattern, jd_low))
        jd_bs = bool(re.search(bs_pattern, jd_low))

        if not (jd_phd or jd_ms or jd_bs):
            return None

        if jd_phd:
            if res_phd:
                return 1.0
            if res_ms:
                return 0.8
            if res_bs:
                return 0.5
            return 0.4
        elif jd_ms:
            if res_phd or res_ms:
                return 1.0
            if res_bs:
                return 0.85
            return 0.6
        elif jd_bs:
            if res_phd or res_ms or res_bs:
                return 1.0
            return 0.7

        return 1.0
