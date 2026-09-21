import numpy as np
import pytest

from rag_assistant.embeddings import VectorStoreManager


def test_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    sim = VectorStoreManager.cosine_similarity(v1, v2)
    assert abs(sim - 1.0) < 1e-5

    v3 = np.array([0.0, 1.0, 0.0])
    sim_ortho = VectorStoreManager.cosine_similarity(v1, v3)
    assert abs(sim_ortho - 0.0) < 1e-5


def test_sha256_fallback_encode():
    vm = VectorStoreManager(allow_fallback=True)
    vm.model = None  # Force SHA-256 fallback
    vm.model_used = "fallback-sha256-hashing"

    vecs = vm.encode(["Python PyTorch RAG", "ROS 2 Gazebo"])
    assert vecs.shape == (2, 384)
    assert abs(float(np.linalg.norm(vecs[0])) - 1.0) < 1e-4


def test_disallow_fallback_raises_error():
    # Invalid model name with fallback disabled must raise RuntimeError
    with pytest.raises(RuntimeError):
        VectorStoreManager(model_name="non_existent_transformer_model_xyz", allow_fallback=False)
