import hashlib
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages neural transformer embeddings and deterministic fallback feature encoding
    for semantic retrieval over resumes and job requirements.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", allow_fallback: bool = False):
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.model = None
        self.model_used = model_name
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.model_used = self.model_name
            logger.info(f"Successfully loaded transformer model: {self.model_name}")
        except Exception as e:
            self.model = None
            if not self.allow_fallback:
                logger.error(
                    f"Failed to load sentence-transformer model '{self.model_name}': {e}. "
                    "Fallback is disabled by default. Pass allow_fallback=True or --allow-fallback "
                    "to explicitly allow SHA-256 feature encoding."
                )
                raise RuntimeError(
                    f"Transformer model loading failed for '{self.model_name}'. "
                    "To enable stable SHA-256 feature encoding fallback, run with --allow-fallback."
                ) from e

            logger.warning(
                f"[EXPLICIT FALLBACK ALLOWED] Could not load model '{self.model_name}'. "
                "Switching to stable SHA-256 HashingVectorizer feature encoding."
            )
            self.model_used = "fallback-sha256-hashing"

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for input text strings.
        Returns L2-normalized float32 numpy array of shape (N, dim).
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        clean_texts = [str(t) if t is not None else "" for t in texts]

        if self.model is not None:
            embeddings = self.model.encode(
                clean_texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
            )
            return embeddings.astype(np.float32)

        # Stable SHA-256 deterministic fallback encoding
        return self._sha256_fallback_encode(clean_texts)

    def _sha256_fallback_encode(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """
        Deterministic, stable feature hashing encoder using SHA-256 digest hashing.
        Replaces un-seeded built-in hash() with cross-platform reproducible hashing.
        """
        embeddings = []
        for t in texts:
            words = t.lower().split()
            v = np.zeros(dim, dtype=np.float32)
            for w in words:
                # Stable 32-bit SHA-256 hash integer modulo dimension
                digest = hashlib.sha256(w.encode("utf-8")).hexdigest()
                idx = int(digest, 16) % dim
                v[idx] += 1.0

            # L2 Normalization
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            embeddings.append(v)

        return np.array(embeddings, dtype=np.float32)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates cosine similarity score between two vector embeddings.
        """
        if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        v1 = vec1.flatten()
        v2 = vec2.flatten()

        dot = float(np.dot(v1, v2))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm1 * norm2)))
