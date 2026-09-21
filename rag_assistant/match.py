import argparse
import json
import logging
import os
import re
import sys
from .parser import DocumentParser
from .embeddings import VectorStoreManager
from .vector_store import PersistentFAISSVectorStore
from .retrieval import SemanticJobRetriever
from .skill_extraction import CategorizedSkillExtractor
from .ranking import ConfigurableJobRanker
from .recommendation import GroundedRecommendationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Match candidate resume against indexed FAISS job corpus.")
    parser.add_argument("--resume", type=str, required=True, help="Path to resume document (.pdf, .docx, .txt).")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Directory containing persistent FAISS index.")
    parser.add_argument("--config", type=str, default="config/scoring.yaml", help="Path to scoring config YAML.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top job chunks to retrieve.")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow SHA-256 fallback if transformer model fails to load.")

    args = parser.parse_args()

    # 1. Parse Resume Document
    logger.info(f"Parsing resume document '{args.resume}'...")
    resume_data = DocumentParser.parse_file(args.resume)
    resume_text = resume_data["clean_text"]

    # 2. Load Persistent FAISS Vector Store
    logger.info(f"Loading persistent FAISS vector store from '{args.artifacts}'...")
    vstore = PersistentFAISSVectorStore()
    try:
        vstore.load(args.artifacts)
    except Exception as e:
        logger.error(f"Could not load vector store from '{args.artifacts}': {e}")
        logger.info("Please run 'python -m rag_assistant.index_jobs' first to index the job corpus.")
        sys.exit(1)

    embedder = VectorStoreManager(allow_fallback=args.allow_fallback)
    retriever = SemanticJobRetriever(vstore, embedder)

    # 3. Retrieve Semantically Relevant Chunks
    logger.info(f"Retrieving top {args.top_k} semantically matching job chunks...")
    retrieved_chunks = retriever.retrieve_top_chunks(resume_text, top_k=args.top_k)

    if not retrieved_chunks:
        logger.warning("No matching job chunks found in index.")
        sys.exit(0)

    # Group by Job ID
    grouped = retriever.group_chunks_by_job(retrieved_chunks)
    ranker = ConfigurableJobRanker(config_path=args.config)
    rec_engine = GroundedRecommendationEngine()

    results = []
    for job_id, chunks in grouped.items():
        first_chunk = chunks[0]
        job_title = first_chunk.get("title", "Target Role")
        company = first_chunk.get("company", "Company")

        # Combine retrieved chunk text for job representation
        job_text = " ".join([c["text"] for c in chunks])
        extracted_jd_skills = list(CategorizedSkillExtractor.extract_categorized_skills(job_text).keys())

        # Compare Skills
        skill_comp = CategorizedSkillExtractor.compare_skills(
            resume_text=resume_text,
            job_required_skills=extracted_jd_skills,
            job_preferred_skills=[],
            job_description_text=job_text
        )

        # Semantic responsibility score
        chunk_sim = max([c["similarity_score"] for c in chunks], default=0.0)

        # Calculate 5-Component Matrix Score
        match_metrics = ranker.compute_composite_score(
            semantic_responsibility_sim=chunk_sim,
            required_skill_coverage=skill_comp["required_coverage"],
            preferred_skill_coverage=skill_comp["preferred_coverage"],
            resume_text=resume_text,
            job={"title": job_title, "description": job_text}
        )

        # Generate Grounded Recommendations with Citations
        recs = rec_engine.generate_recommendations(
            candidate_name=resume_data["file_name"],
            retrieved_chunks=chunks,
            matching_skills=skill_comp["matching_skills"],
            missing_required_skills=skill_comp["missing_required_skills"],
            missing_preferred_skills=skill_comp["missing_preferred_skills"],
            overall_score=match_metrics["overall_match_score"]
        )

        results.append({
            "job_id": job_id,
            "job_title": job_title,
            "company": company,
            "match_metrics": match_metrics,
            "skill_comparison": skill_comp,
            "recommendations": recs,
            "retrieved_chunks": chunks
        })

    # Sort results by overall match score descending
    results.sort(key=lambda x: x["match_metrics"]["overall_match_score"], reverse=True)

    # Print summary output
    print("\n========================================================")
    print(f"   RAG CAREER MATCHING ANALYSIS REPORT for {resume_data['file_name']}")
    print("========================================================\n")
    print(f"Model Encoder Used: {embedder.model_used}")
    print(f"Total Jobs Evaluated: {len(results)}\n")

    for idx, res in enumerate(results, 1):
        print(f"--- #{idx} {res['job_title']} @ {res['company']} ---")
        print(f"Overall Score: {res['match_metrics']['overall_match_score']}%")
        print(f"Breakdown: {json.dumps(res['match_metrics']['component_breakdown'], indent=2)}")
        print(f"Matching Skills ({len(res['skill_comparison']['matching_skills'])}): {[m['skill'] for m in res['skill_comparison']['matching_skills']]}")
        print(f"Missing Required Skills ({len(res['skill_comparison']['missing_required_skills'])}): {[m['skill'] for m in res['skill_comparison']['missing_required_skills']]}")
        print("\nTop Grounded Recommendations & Citations:")
        for rec in res["recommendations"]["resume_recommendations"][:2]:
            print(f"  • [{rec['type']}] {rec['action']} (Citing: {rec['cited_job_chunk']})")
        print("-" * 56 + "\n")


if __name__ == "__main__":
    main()
