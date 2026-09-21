import argparse
import logging
import os
import sys
from .ingestion import JobCorpusIngestor
from .chunking import SectionAwareChunker
from .embeddings import VectorStoreManager
from .vector_store import PersistentFAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch index job corpus into persistent FAISS vector store.")
    parser.add_argument("--input", type=str, default="data/jobs", help="Path to job JSON files directory.")
    parser.add_argument("--output", type=str, default="artifacts", help="Directory to save persistent index files.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Transformer model name.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow fallback SHA-256 encoding if model fails to load.")

    args = parser.parse_args()

    logger.info(f"Ingesting jobs from '{args.input}'...")
    jobs = JobCorpusIngestor.load_corpus(args.input)

    if not jobs:
        logger.error(f"No job records found in {args.input}. Indexing aborted.")
        sys.exit(1)

    chunker = SectionAwareChunker()
    all_chunks = []
    for job in jobs:
        chunks = chunker.chunk_job_description(job)
        all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} chunks from {len(jobs)} jobs.")

    embedder = VectorStoreManager(model_name=args.model, allow_fallback=args.allow_fallback)
    texts = [c["text"] for c in all_chunks]

    logger.info(f"Encoding {len(texts)} text chunks using {embedder.model_used}...")
    embeddings = embedder.encode(texts)

    vstore = PersistentFAISSVectorStore(dimension=embeddings.shape[1] if len(embeddings) > 0 else 384)
    vstore.add_chunks(all_chunks, embeddings)
    vstore.save(args.output)

    logger.info(f"Successfully built persistent FAISS vector store in '{args.output}'.")


if __name__ == "__main__":
    main()
