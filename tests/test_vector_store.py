import os
import pytest
import numpy as np
from rag_assistant.vector_store import PersistentFAISSVectorStore


def test_vector_store_add_and_search():
    vstore = PersistentFAISSVectorStore(dimension=4)
    chunks = [
        {"chunk_id": "c1", "text": "RAG and FAISS vector indexing"},
        {"chunk_id": "c2", "text": "ROS 2 TurtleBot Gazebo navigation"}
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    vstore.add_chunks(chunks, embeddings)

    # Search for vector close to c1
    query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    results = vstore.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["similarity_score"] > 0.8


def test_vector_store_persistence(tmp_path):
    artifacts = tmp_path / "artifacts"
    vstore = PersistentFAISSVectorStore(dimension=4)
    chunks = [{"chunk_id": "c1", "text": "Test chunk"}]
    embeddings = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    vstore.add_chunks(chunks, embeddings)
    vstore.save(str(artifacts))

    assert os.path.exists(artifacts / "job_metadata.json")

    # Load into new vector store
    vstore_loaded = PersistentFAISSVectorStore(dimension=4)
    vstore_loaded.load(str(artifacts))

    assert len(vstore_loaded.metadata_store) == 1
    assert vstore_loaded.metadata_store[0]["chunk_id"] == "c1"
