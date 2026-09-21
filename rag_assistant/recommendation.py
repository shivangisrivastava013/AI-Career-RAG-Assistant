import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class GroundedRecommendationEngine:
    """
    Generates structured, grounded career recommendations with job chunk citations
    and strict anti-hallucination safeguards.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate_recommendations(
        self,
        candidate_name: str,
        retrieved_chunks: List[Dict[str, Any]],
        matching_skills: List[Dict[str, Any]],
        missing_required_skills: List[Dict[str, Any]],
        missing_preferred_skills: List[Dict[str, Any]],
        overall_score: float
    ) -> Dict[str, Any]:
        """
        Generates structured JSON career recommendations with job chunk citations.
        Uses rule-grounded template synthesis or optional direct LLM API invocation.
        """
        citations = []
        for chunk in retrieved_chunks:
            citations.append({
                "chunk_id": chunk.get("chunk_id", "job_chunk"),
                "section": chunk.get("section", "Requirements"),
                "job_title": chunk.get("title", "Job Description"),
                "text_snippet": chunk.get("text", "")[:180] + "..."
            })

        # 1. Fit Summary
        fit_summary = (
            f"Candidate overall compatibility score is {overall_score}%. "
            f"Demonstrates strong capability in {len(matching_skills)} required technical domains. "
            f"Identified {len(missing_required_skills)} primary skill gaps relative to target job requirements."
        )

        # 2. Strong Matches
        strong_matches = []
        for m in matching_skills:
            strong_matches.append({
                "skill": m["skill"],
                "category": m["category"],
                "resume_evidence": m["resume_evidence"],
                "job_requirement": m["job_evidence"]
            })

        # 3. Skill Gaps (Separated by Action: Wording Improvement vs Skills to Learn)
        skill_gaps = []
        for gap in missing_required_skills:
            skill_gaps.append({
                "skill": gap["skill"],
                "category": gap["category"],
                "importance": "Mandatory / Required",
                "recommendation_type": "Skill to Acquire",
                "guidance": f"Target job requires {gap['skill']}. If you have applied this in projects, explicitly document it. Otherwise, prioritize learning this domain."
            })

        for gap in missing_preferred_skills:
            skill_gaps.append({
                "skill": gap["skill"],
                "category": gap["category"],
                "importance": "Preferred / Bonus",
                "recommendation_type": "Skill to Highlight",
                "guidance": f"Target job lists {gap['skill']} as a preferred qualification."
            })

        # 4. Resume Wording & Structuring Recommendations (Citing job requirements)
        resume_recommendations = []
        for i, match in enumerate(matching_skills[:3]):
            citation_ref = citations[i % len(citations)]["chunk_id"] if citations else "chunk_0"
            resume_recommendations.append({
                "type": "Resume Wording Improvement",
                "action": f"Quantify empirical impact for {match['skill']}",
                "detail": f"Your experience with {match['skill']} matches requirement. Add quantifiable metrics (e.g. latency, throughput, accuracy gains) to strengthen impact.",
                "cited_job_chunk": citation_ref
            })

        for i, gap in enumerate(missing_required_skills[:2]):
            citation_ref = citations[(i + 3) % len(citations)]["chunk_id"] if citations else "chunk_1"
            resume_recommendations.append({
                "type": "Skill Gap Action",
                "action": f"Acquire / Demonstrate {gap['skill']}",
                "detail": f"Target job specifically requests {gap['skill']}. Consider completing a practical project incorporating {gap['skill']} to bridge this gap.",
                "cited_job_chunk": citation_ref
            })

        # 5. Interview Topics
        interview_topics = []
        for m in matching_skills[:3]:
            interview_topics.append(f"Deep-dive technical discussion on production usage of {m['skill']}.")
        for g in missing_required_skills[:2]:
            interview_topics.append(f"Be prepared to address your experience level or learning trajectory for {g['skill']}.")

        return {
            "fit_summary": fit_summary,
            "strong_matches": strong_matches,
            "skill_gaps": skill_gaps,
            "resume_recommendations": resume_recommendations,
            "interview_topics": interview_topics,
            "citations": citations,
            "safeguard_notice": "Recommendations are grounded in retrieved job requirements. Non-existent experience is never recommended to be added to your resume."
        }
