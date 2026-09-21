import os

import numpy as np
import pytest

from rag_assistant.vector_store import PersistentFAISSVectorStore


def test_vector_store_add_and_search():
    vstore = PersistentFAISSVectorStore(dimension=4, encoder_model="test_encoder")
    chunks = [
        {"chunk_id": "c1", "text": "RAG and FAISS vector indexing"},
        {"chunk_id": "c2", "text": "ROS 2 TurtleBot Gazebo navigation"},
    ]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    vstore.add_chunks(chunks, embeddings)

    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = vstore.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity_score"] > 0.8


def test_vector_store_persistence(tmp_path):
    artifacts = tmp_path / "artifacts"
    vstore = PersistentFAISSVectorStore(dimension=4, encoder_model="model_a")
    chunks = [{"chunk_id": "c1", "text": "Test chunk"}]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    vstore.add_chunks(chunks, embeddings)
    vstore.save(str(artifacts))

    assert os.path.exists(artifacts / "job_metadata.json")
    assert os.path.exists(artifacts / "index_config.json")

    vstore_loaded = PersistentFAISSVectorStore(dimension=4)
    vstore_loaded.load(str(artifacts), expected_encoder="model_a")

    assert len(vstore_loaded.metadata_store) == 1
    assert vstore_loaded.metadata_store[0]["chunk_id"] == "c1"


def test_encoder_mismatch_raises_error(tmp_path):
    artifacts = tmp_path / "artifacts"
    vstore = PersistentFAISSVectorStore(dimension=4, encoder_model="model_a")
    chunks = [{"chunk_id": "c1", "text": "Test chunk"}]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vstore.add_chunks(chunks, embeddings)
    vstore.save(str(artifacts))

    vstore_loaded = PersistentFAISSVectorStore(dimension=4)
    with pytest.raises(ValueError):
        vstore_loaded.load(str(artifacts), expected_encoder="model_b")
