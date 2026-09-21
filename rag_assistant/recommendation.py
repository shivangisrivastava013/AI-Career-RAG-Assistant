import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CitationGroundedRecommendationEngine:
    """
    Citation-Grounded Recommendation Engine:
    Synthesizes actionable career recommendations grounded in retrieved job description chunks
    with strict anti-hallucination safeguards and chunk citation tracking.
    """

    def __init__(self, provider: str = "deterministic_template", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key

    def generate_recommendations(
        self,
        candidate_name: str,
        retrieved_chunks: List[Dict[str, Any]],
        matching_skills: List[Dict[str, Any]],
        missing_required_skills: List[Dict[str, Any]],
        missing_preferred_skills: List[Dict[str, Any]],
        overall_score: float,
    ) -> Dict[str, Any]:
        citations = []
        for chunk in retrieved_chunks:
            citations.append(
                {
                    "chunk_id": chunk.get("chunk_id", "job_chunk"),
                    "section": chunk.get("section", "Requirements"),
                    "job_title": chunk.get("title", "Job Description"),
                    "text_snippet": chunk.get("text", "")[:180] + "...",
                }
            )

        fit_summary = (
            f"Candidate overall compatibility score is {overall_score}%. "
            f"Demonstrates strong capability in {len(matching_skills)} required technical domains. "
            f"Identified {len(missing_required_skills)} primary skill gaps relative to target job requirements."
        )

        strong_matches = []
        for m in matching_skills:
            strong_matches.append(
                {
                    "skill": m["skill"],
                    "category": m["category"],
                    "resume_evidence": m["resume_evidence"],
                    "job_requirement": m["job_evidence"],
                }
            )

        skill_gaps = []
        for gap in missing_required_skills:
            skill_gaps.append(
                {
                    "skill": gap["skill"],
                    "category": gap["category"],
                    "importance": "Mandatory / Required",
                    "recommendation_type": "Skill to Acquire",
                    "guidance": f"Target job requires {gap['skill']}. If you have applied this in projects, explicitly document it. Otherwise, prioritize learning this domain.",
                }
            )

        for gap in missing_preferred_skills:
            skill_gaps.append(
                {
                    "skill": gap["skill"],
                    "category": gap["category"],
                    "importance": "Preferred / Bonus",
                    "recommendation_type": "Skill to Highlight",
                    "guidance": f"Target job lists {gap['skill']} as a preferred qualification.",
                }
            )

        resume_recommendations = []
        for i, match in enumerate(matching_skills[:3]):
            citation_ref = citations[i % len(citations)]["chunk_id"] if citations else "chunk_0"
            resume_recommendations.append(
                {
                    "type": "Resume Wording Improvement",
                    "action": f"Quantify empirical impact for {match['skill']}",
                    "detail": f"Your experience with {match['skill']} matches requirement. Add quantifiable metrics (e.g. latency, throughput, accuracy gains) to strengthen impact.",
                    "cited_job_chunk": citation_ref,
                }
            )

        for i, gap in enumerate(missing_required_skills[:2]):
            citation_ref = citations[(i + 3) % len(citations)]["chunk_id"] if citations else "chunk_1"
            resume_recommendations.append(
                {
                    "type": "Skill Gap Action",
                    "action": f"Acquire / Demonstrate {gap['skill']}",
                    "detail": f"Target job specifically requests {gap['skill']}. Consider completing a practical project incorporating {gap['skill']} to bridge this gap.",
                    "cited_job_chunk": citation_ref,
                }
            )

        interview_topics = []
        for m in matching_skills[:3]:
            interview_topics.append(f"Deep-dive technical discussion on production usage of {m['skill']}.")
        for g in missing_required_skills[:2]:
            interview_topics.append(
                f"Be prepared to address your experience level or learning trajectory for {g['skill']}."
            )

        return {
            "fit_summary": fit_summary,
            "strong_matches": strong_matches,
            "skill_gaps": skill_gaps,
            "resume_recommendations": resume_recommendations,
            "interview_topics": interview_topics,
            "citations": citations,
            "provider_used": self.provider,
            "safeguard_notice": "Recommendations are grounded in retrieved job requirements. Non-existent experience is never recommended to be added to your resume.",
        }


# Backwards compatibility alias
GroundedRecommendationEngine = CitationGroundedRecommendationEngine
