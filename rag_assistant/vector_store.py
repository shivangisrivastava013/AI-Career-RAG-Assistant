import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class PersistentFAISSVectorStore:
    """
    Manages persistent FAISS vector indexing (IndexFlatIP / IndexFlatL2)
    and JSON metadata storage for semantic job retrieval.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.faiss_index = None
        self.metadata_store: List[Dict[str, Any]] = []
        self.use_faiss = False
        self._init_faiss()

    def _init_faiss(self):
        try:
            import faiss
            # Inner-product index for normalized cosine similarity search
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.use_faiss = True
            logger.info("FAISS vector index initialized successfully.")
        except Exception as e:
            logger.warning(f"FAISS not available ({e}). Using NumPy inner-product vector indexing.")
            self.use_faiss = False
            self.vectors = np.zeros((0, self.dimension), dtype=np.float32)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Adds text chunks and normalized embeddings to the vector store.
        """
        if len(chunks) == 0 or len(embeddings) == 0:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) mismatch with embeddings count ({len(embeddings)})")

        # Ensure float32 and L2 normalization
        vecs = embeddings.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        vecs = vecs / norms

        if self.use_faiss:
            import faiss
            self.faiss_index.add(vecs)
        else:
            if len(self.vectors) == 0:
                self.vectors = vecs
            else:
                self.vectors = np.vstack([self.vectors, vecs])

        self.metadata_store.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store. Total chunks: {len(self.metadata_store)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches top-k most similar chunks for a given query vector.
        """
        if len(self.metadata_store) == 0:
            return []

        q_vec = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        top_k = min(top_k, len(self.metadata_store))

        if self.use_faiss:
            distances, indices = self.faiss_index.search(q_vec, top_k)
            scores = distances[0]
            idxs = indices[0]
        else:
            sims = np.dot(self.vectors, q_vec.T).flatten()
            idxs = np.argsort(sims)[::-1][:top_k]
            scores = sims[idxs]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, idxs)):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            item = dict(self.metadata_store[idx])
            item["similarity_score"] = float(max(0.0, min(1.0, score)))
            item["rank"] = rank + 1
            results.append(item)

        return results

    def save(self, artifacts_dir: str):
        """
        Persists FAISS index, metadata JSON, and config file to artifacts directory.
        """
        os.makedirs(artifacts_dir, exist_ok=True)

        faiss_path = os.path.join(artifacts_dir, "jobs.faiss")
        metadata_path = os.path.join(artifacts_dir, "job_metadata.json")
        config_path = os.path.join(artifacts_dir, "index_config.json")

        if self.use_faiss:
            import faiss
            faiss.write_index(self.faiss_index, faiss_path)
        else:
            np.save(faiss_path + ".npy", self.vectors)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)

        config = {
            "dimension": self.dimension,
            "total_chunks": len(self.metadata_store),
            "use_faiss": self.use_faiss
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved persistent vector store to '{artifacts_dir}'")

    def load(self, artifacts_dir: str):
        """
        Loads persistent FAISS index and metadata JSON from artifacts directory.
        """
        faiss_path = os.path.join(artifacts_dir, "jobs.faiss")
        metadata_path = os.path.join(artifacts_dir, "job_metadata.json")
        config_path = os.path.join(artifacts_dir, "index_config.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file missing at: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.dimension = cfg.get("dimension", 384)

        if self.use_faiss and os.path.exists(faiss_path):
            import faiss
            self.faiss_index = faiss.read_index(faiss_path)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors.")
        elif os.path.exists(faiss_path + ".npy"):
            self.vectors = np.load(faiss_path + ".npy")
            self.use_faiss = False
            logger.info(f"Loaded NumPy vectors with shape {self.vectors.shape}")
        else:
            logger.warning(f"No FAISS index found at {faiss_path}. Initialized empty vector store.")
