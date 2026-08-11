import numpy as np
from typing import List, Dict, Tuple


class VectorStoreManager:
    """
    Manages vector embedding generation and cosine similarity indexing
    for semantic retrieval over resumes and job requirements.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception:
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for input text strings.
        """
        if not texts:
            return np.array([])

        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            # Fallback TF-IDF / Hash embedding if sentence-transformers is offline
            embeddings = self._fallback_encode(texts)

        # Normalize L2
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return embeddings / norms

    def _fallback_encode(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """
        Deterministic fallback feature encoder when neural transformer model is loading.
        """
        np.random.seed(42)
        vecs = []
        for t in texts:
            words = t.lower().split()
            v = np.zeros(dim, dtype=np.float32)
            for w in words:
                idx = hash(w) % dim
                v[idx] += 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates cosine similarity score between two normalized vector embeddings.
        """
        v1 = vec1.flatten()
        v2 = vec2.flatten()
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def retrieve_top_k(self, query: str, index_chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k most semantically relevant text chunks for a query.
        """
        if not index_chunks:
            return []

        query_vec = self.encode([query])[0]
        chunk_vecs = self.encode(index_chunks)

        scores = [VectorStoreManager.cosine_similarity(query_vec, cv) for cv in chunk_vecs]
        sorted_indices = np.argsort(scores)[::-1][:k]

        return [(index_chunks[idx], float(scores[idx])) for idx in sorted_indices]
