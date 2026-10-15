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


class KeywordJobRetriever:
    """
    Lexical keyword/token-overlap baseline retriever used for benchmark ablation.
    Computes term-frequency overlap between resume query and indexed job chunks.
    """

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve_candidate_jobs(
        self, query_text: str, top_k_chunks: int = 40, max_candidate_jobs: int = 15
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not query_text:
            return {}

        import re

        query_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.\#\+]{2,}\b", query_text.lower()))
        if not query_tokens:
            return {}

        scored = []
        for chunk in self.vector_store.metadata_store:
            chunk_text = chunk.get("text", "").lower()
            chunk_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.\#\+]{2,}\b", chunk_text))
            if not chunk_tokens:
                continue

            overlap = len(query_tokens.intersection(chunk_tokens))
            if overlap > 0:
                score = overlap / (len(query_tokens) ** 0.5 * len(chunk_tokens) ** 0.5)
                c_copy = dict(chunk)
                c_copy["similarity_score"] = round(float(score), 4)
                c_copy["model_used"] = "keyword-lexical-baseline"
                scored.append((score, c_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored[:top_k_chunks]]

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for c in top_chunks:
            job_id = c.get("job_id", "default_job")
            if job_id not in grouped:
                if len(grouped) >= max_candidate_jobs:
                    continue
                grouped[job_id] = []
            grouped[job_id].append(c)

        logger.info(f"[Keyword Ablation] Retrieved {len(top_chunks)} chunks across {len(grouped)} unique jobs.")
        return grouped
