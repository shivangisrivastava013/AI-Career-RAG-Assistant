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
        job=job
    )

    assert 0.0 <= res["overall_match_score"] <= 100.0
    assert 0.0 <= res["composite_score"] <= 1.0
    assert "required_skill_coverage" in res["component_breakdown"]


def test_custom_weights(tmp_path):
    cfg_file = tmp_path / "custom_scoring.yaml"
    cfg_file.write_text("""
scoring_weights:
  required_skills: 0.50
  responsibilities: 0.30
  experience_level: 0.10
  education_level: 0.05
  preferred_skills: 0.05
""")

    ranker = ConfigurableJobRanker(config_path=str(cfg_file))
    assert abs(ranker.weights["required_skills"] - 0.50) < 1e-4
