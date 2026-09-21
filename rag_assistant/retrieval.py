import logging
from typing import List, Dict, Any
from .embeddings import VectorStoreManager
from .vector_store import PersistentFAISSVectorStore

logger = logging.getLogger(__name__)


class SemanticJobRetriever:
    """
    Retrieves most relevant job chunks and aggregates context per job description.
    """

    def __init__(self, vector_store: PersistentFAISSVectorStore, embedder: VectorStoreManager):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_top_chunks(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most semantically relevant job chunks for a resume query string.
        """
        if not query_text:
            return []

        query_vec = self.embedder.encode([query_text])[0]
        chunks = self.vector_store.search(query_vec, top_k=top_k)

        # Attach model used tag
        for c in chunks:
            c["model_used"] = self.embedder.model_used

        return chunks

    def group_chunks_by_job(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups retrieved chunks by job_id.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            job_id = c.get("job_id", "default_job")
            if job_id not in grouped:
                grouped[job_id] = []
            grouped[job_id].append(c)
        return grouped
