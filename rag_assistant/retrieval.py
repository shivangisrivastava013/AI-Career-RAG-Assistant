import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SemanticJobRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Retrieve top-N candidate chunks across corpus and group by job_id (up to 15 candidate jobs).
    Stage 2: Pass candidate jobs to structured scoring matrix and select final top-K jobs.
    """

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_candidate_jobs(
        self, query_text: str, top_k_chunks: int = 40, max_candidate_jobs: int = 15
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves top candidate chunks and groups them into candidate job descriptions.
        """
        if not query_text:
            return {}

        query_vec = self.embedder.encode([query_text])[0]
        retrieved_chunks = self.vector_store.search(query_vec, top_k=top_k_chunks)

        for c in retrieved_chunks:
            c["model_used"] = self.embedder.model_used

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for c in retrieved_chunks:
            job_id = c.get("job_id", "default_job")
            if job_id not in grouped:
                if len(grouped) >= max_candidate_jobs:
                    continue
                grouped[job_id] = []
            grouped[job_id].append(c)

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks across {len(grouped)} unique candidate jobs.")
        return grouped
