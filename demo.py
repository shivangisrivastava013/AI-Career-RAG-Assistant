import argparse
import os
import sys

from rag_assistant.embeddings import VectorStoreManager
from rag_assistant.matcher import CareerRAGMatcher
from rag_assistant.parser import DocumentParser
from rag_assistant.ranking import ConfigurableJobRanker
from rag_assistant.recommendation import CitationGroundedRecommendationEngine
from rag_assistant.retrieval import SemanticJobRetriever
from rag_assistant.skill_extraction import CategorizedSkillExtractor
from rag_assistant.vector_store import PersistentFAISSVectorStore


def run_single_job_match(resume_path: str, job_path: str, allow_fallback: bool = True):
    print("\n" + "=" * 64)
    print("   [1] DIRECT PAIRWISE RESUME & JOB SPEC MATCHING")
    print("=" * 64)

    resume_text = DocumentParser.load_file(resume_path)
    job_text = DocumentParser.load_file(job_path)

    embedder = VectorStoreManager(allow_fallback=allow_fallback)
    matcher = CareerRAGMatcher(vector_store=embedder)
    results = matcher.evaluate_match(resume_text, job_text)

    print(f"[*] Overall Match Score:        {results['overall_match_percentage']}%")
    print(f"[*] Semantic Similarity:        {results['semantic_similarity_score']}")
    print(f"[*] Skill Coverage Ratio:       {results['skill_coverage_ratio'] * 100:.1f}%")
    print(f"[+] Matching Skills Found ({len(results['matching_skills'])}): {', '.join(results['matching_skills'])}")
    print(
        f"[!] Missing Skills ({len(results['missing_skills'])}):        {', '.join(results['missing_skills']) if results['missing_skills'] else 'None'}"
    )
    print("\n[Feedback] Grounded Recommendation:")
    print(f"   {results['recommendation']}")


def run_corpus_rag_walkthrough(resume_path: str, artifacts_dir: str = "artifacts", allow_fallback: bool = True):
    print("\n" + "=" * 64)
    print("   [2] END-TO-END RAG CORPUS RETRIEVAL & GROUNDED RANKING")
    print("       (Evaluated against Synthetic 50-Job Corpus)")
    print("=" * 64)

    if not os.path.exists(os.path.join(artifacts_dir, "job_metadata.json")):
        print(f"[!] Artifacts directory '{artifacts_dir}' not found or empty.")
        print("[!] Rebuilding index from data/jobs...")
        from rag_assistant.index_jobs import build_index

        build_index("data/jobs", artifacts_dir, allow_fallback=allow_fallback)

    resume_data = DocumentParser.parse_file(resume_path)
    resume_text = resume_data["clean_text"]

    embedder = VectorStoreManager(allow_fallback=allow_fallback)
    vstore = PersistentFAISSVectorStore()
    vstore.load(artifacts_dir, expected_encoder=embedder.model_used)

    retriever = SemanticJobRetriever(vstore, embedder)
    candidate_jobs = retriever.retrieve_candidate_jobs(resume_text, top_k_chunks=40, max_candidate_jobs=10)

    ranker = ConfigurableJobRanker(config_path="config/scoring.yaml")
    rec_engine = CitationGroundedRecommendationEngine()

    ranked_jobs = []
    for job_id, chunks in candidate_jobs.items():
        first_c = chunks[0]
        title = first_c.get("title", "Target Role")
        company = first_c.get("company", "Company")
        req_skills = list(first_c.get("required_skills", []))
        pref_skills = list(first_c.get("preferred_skills", []))
        job_text = " ".join([c["text"] for c in chunks])

        skill_comp = CategorizedSkillExtractor.compare_skills(
            resume_text=resume_text,
            job_required_skills=req_skills,
            job_preferred_skills=pref_skills,
            job_description_text=job_text,
        )

        chunk_sim = max([c["similarity_score"] for c in chunks], default=0.0)
        metrics = ranker.compute_composite_score(
            semantic_responsibility_sim=chunk_sim,
            required_skill_coverage=skill_comp["required_coverage"],
            preferred_skill_coverage=skill_comp["preferred_coverage"] if pref_skills else None,
            resume_text=resume_text,
            job={"title": title, "description": job_text},
        )

        recs = rec_engine.generate_recommendations(
            candidate_name=resume_data["file_name"],
            retrieved_chunks=chunks,
            matching_skills=skill_comp["matching_skills"],
            missing_required_skills=skill_comp["missing_required_skills"],
            missing_preferred_skills=skill_comp["missing_preferred_skills"],
            overall_score=metrics["overall_match_score"],
        )

        ranked_jobs.append(
            {
                "job_id": job_id,
                "title": title,
                "company": company,
                "metrics": metrics,
                "skill_comp": skill_comp,
                "recs": recs,
            }
        )

    ranked_jobs.sort(key=lambda x: x["metrics"]["overall_match_score"], reverse=True)

    print(f"\n[*] Uploaded Candidate Resume: {resume_data['file_name']}")
    print(f"[*] Retrieved Candidate Roles: {len(ranked_jobs)} (Displaying Top 3)\n")

    for idx, r in enumerate(ranked_jobs[:3], 1):
        print(f"--- Top Match #{idx}: {r['title']} @ {r['company']} ---")
        print(f"    Overall Match Score: {r['metrics']['overall_match_score']}%")
        print(f"    Required Skills Coverage: {r['metrics']['component_breakdown']['required_skills']}%")
        print(f"    Matching Skills ({len(r['skill_comp']['matching_skills'])}): {[m['skill'] for m in r['skill_comp']['matching_skills']]}")
        print(
            f"    Missing Required Skills ({len(r['skill_comp']['missing_required_skills'])}): {[m['skill'] for m in r['skill_comp']['missing_required_skills']]}"
        )
        print("    Grounded Recommendations:")
        for rec in r["recs"]["resume_recommendations"][:2]:
            print(f"      - [{rec['type']}] {rec['action']} (Citing: {rec['cited_job_chunk']})")
        print()


def main():
    parser = argparse.ArgumentParser(description="AI Career RAG Assistant CLI Walkthrough Demo.")
    parser.add_argument("--resume", type=str, default="data/sample_resume.txt", help="Path to resume.")
    parser.add_argument("--job", type=str, default="data/sample_job.txt", help="Path to target job description.")
    parser.add_argument("--allow-fallback", action="store_true", default=True, help="Allow SHA-256 fallback if transformer is absent.")
    args = parser.parse_args()

    print("[+] Initializing AI Career RAG Assistant...")
    if not os.path.exists(args.resume) or not os.path.exists(args.job):
        print(f"[!] Input files not found: {args.resume} or {args.job}")
        sys.exit(1)

    run_single_job_match(args.resume, args.job, allow_fallback=args.allow_fallback)
    run_corpus_rag_walkthrough(args.resume, allow_fallback=args.allow_fallback)
    print("=" * 64)
    print("[+] Demo walkthrough completed successfully.")
    print("=" * 64)


if __name__ == "__main__":
    main()
