from rag_assistant.recommendation import GroundedRecommendationEngine


def test_recommendation_groundedness_schema():
    rec_engine = GroundedRecommendationEngine()
    retrieved_chunks = [
        {
            "chunk_id": "job_001_chunk_0",
            "title": "RAG Engineer",
            "section": "Requirements",
            "text": "Responsibilities: Build production RAG vector search pipelines with FAISS and PyTorch.",
        }
    ]
    matching_skills = [
        {
            "skill": "PyTorch",
            "category": "ml_frameworks",
            "resume_evidence": "2 years experience with PyTorch.",
            "job_evidence": "PyTorch required.",
        }
    ]
    missing_required = [
        {"skill": "FAISS", "category": "genai_technologies", "job_evidence": "FAISS vector search required."}
    ]

    recs = rec_engine.generate_recommendations(
        candidate_name="Test Candidate",
        retrieved_chunks=retrieved_chunks,
        matching_skills=matching_skills,
        missing_required_skills=missing_required,
        missing_preferred_skills=[],
        overall_score=85.0,
    )

    assert "fit_summary" in recs
    assert "strong_matches" in recs
    assert "skill_gaps" in recs
    assert "resume_recommendations" in recs
    assert "citations" in recs
    assert len(recs["citations"]) > 0
    assert recs["citations"][0]["chunk_id"] == "job_001_chunk_0"
    assert recs["safeguard_notice"] is not None
