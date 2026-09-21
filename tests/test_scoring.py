import pytest

from rag_assistant.ranking import ConfigurableJobRanker


def test_scoring_matrix_bounds():
    ranker = ConfigurableJobRanker()
    job = {"title": "AI Engineer", "description": "2+ years PyTorch experience. Master's degree required."}
    resume = "MS in AI Candidate at NJIT with 2+ years PyTorch experience."

    res = ranker.compute_composite_score(
        semantic_responsibility_sim=0.85,
        required_skill_coverage=1.0,
        preferred_skill_coverage=0.5,
        resume_text=resume,
        job=job,
    )

    assert 0.0 <= res["overall_match_score"] <= 100.0
    assert 0.0 <= res["composite_score"] <= 1.0
    assert "required_skills" in res["component_breakdown"]


def test_education_word_boundary_regex():
    # Word boundary prevents false matches like "cms" or "ams"
    ranker = ConfigurableJobRanker()
    job = {"description": "Master's degree required"}

    res_ms = ranker._evaluate_education_alignment("MS in Artificial Intelligence", job["description"])
    assert res_ms == 1.0

    res_cms = ranker._evaluate_education_alignment("CMS developer experience", job["description"])
    assert res_cms == 0.6  # No MS match (word boundary prevents matching 'ms' inside 'cms')


def test_dynamic_weight_redistribution():
    ranker = ConfigurableJobRanker()
    job = {"description": "PyTorch role with no experience or education listed."}
    resume = "PyTorch developer."

    res = ranker.compute_composite_score(
        semantic_responsibility_sim=0.80,
        required_skill_coverage=1.0,
        preferred_skill_coverage=None,  # Unavailable component
        resume_text=resume,
        job=job,
    )

    assert res["component_breakdown"]["preferred_skills"] == "insufficient_evidence"
    assert "required_skills" in res["redistributed_weights"]
    assert sum(res["redistributed_weights"].values()) == pytest.approx(1.0)
