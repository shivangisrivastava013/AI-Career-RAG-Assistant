import argparse
import logging
import sys

from .chunking import SectionAwareChunker
from .embeddings import VectorStoreManager
from .ingestion import JobCorpusIngestor
from .vector_store import PersistentFAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_index(
    input_dir: str = "data/jobs",
    output_dir: str = "artifacts",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    allow_fallback: bool = False,
) -> PersistentFAISSVectorStore:
    """
    Ingests job JSON files, chunks descriptions, generates embeddings,
    and saves a persistent FAISS vector store with exact encoder metadata.
    """
    logger.info(f"Ingesting jobs from '{input_dir}'...")
    jobs = JobCorpusIngestor.load_corpus(input_dir)

    if not jobs:
        raise ValueError(f"No job records found in '{input_dir}' for indexing.")

    chunker = SectionAwareChunker()
    all_chunks = []
    for job in jobs:
        all_chunks.extend(chunker.chunk_job_description(job))

    logger.info(f"Generated {len(all_chunks)} chunks from {len(jobs)} jobs.")

    embedder = VectorStoreManager(model_name=model_name, allow_fallback=allow_fallback)
    texts = [chunk["text"] for chunk in all_chunks]

    logger.info(f"Encoding {len(texts)} text chunks using {embedder.model_used}...")
    embeddings = embedder.encode(texts)

    vstore = PersistentFAISSVectorStore(
        dimension=embeddings.shape[1] if len(embeddings) > 0 else 384,
        encoder_model=embedder.model_used,
    )
    vstore.add_chunks(all_chunks, embeddings)
    vstore.save(output_dir)

    logger.info(f"Successfully built persistent FAISS vector store in '{output_dir}'.")
    return vstore


def main():
    parser = argparse.ArgumentParser(description="Batch index job corpus into persistent FAISS vector store.")
    parser.add_argument("--input", type=str, default="data/jobs", help="Path to job JSON files directory.")
    parser.add_argument("--output", type=str, default="artifacts", help="Directory to save persistent index files.")
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Transformer model name."
    )
    parser.add_argument(
        "--allow-fallback", action="store_true", help="Allow fallback SHA-256 encoding if model fails to load."
    )

    args = parser.parse_args()

    try:
        build_index(
            input_dir=args.input,
            output_dir=args.output,
            model_name=args.model,
            allow_fallback=args.allow_fallback,
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
