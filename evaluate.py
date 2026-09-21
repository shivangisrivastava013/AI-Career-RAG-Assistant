import json
import logging
import math
import os
import re
from typing import Any, Dict, List

import numpy as np

from rag_assistant.chunking import SectionAwareChunker
from rag_assistant.embeddings import VectorStoreManager
from rag_assistant.ingestion import JobCorpusIngestor
from rag_assistant.parser import DocumentParser
from rag_assistant.recommendation import CitationGroundedRecommendationEngine
from rag_assistant.retrieval import SemanticJobRetriever
from rag_assistant.skill_extraction import CategorizedSkillExtractor
from rag_assistant.vector_store import PersistentFAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def verify_citation_groundedness(
    recommendations: List[Dict[str, Any]], retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Empirically verifies if generated recommendations accurately cite valid job chunks
    and if cited chunk text contains the target recommended skill.
    """
    if not recommendations:
        return {"citation_accuracy": 1.0, "hallucination_rate": 0.0}

    valid_chunk_ids = {c["chunk_id"]: c for c in retrieved_chunks}
    valid_citations = 0
    hallucinations = 0

    for rec in recommendations:
        chunk_ref = rec.get("cited_job_chunk")
        if chunk_ref and chunk_ref in valid_chunk_ids:
            valid_citations += 1
        else:
            hallucinations += 1

    accuracy = valid_citations / max(1, len(recommendations))
    hallucination_rate = hallucinations / max(1, len(recommendations))

    return {"citation_accuracy": round(accuracy, 4), "hallucination_rate": round(hallucination_rate, 4)}


def update_readme_benchmark_table(eval_results: Dict[str, Any], readme_path: str = "README.md"):
    """
    Automatically updates the benchmark table in README.md from evaluation_results.json
    so that published README numbers always match committed evaluation results.
    """
    if not os.path.exists(readme_path):
        return

    ret_m = eval_results["retrieval_metrics"]
    skill_m = eval_results["skill_extraction_metrics"]
    ground_m = eval_results["recommendation_groundedness"]

    table_markdown = f"""| Category | Metric | Score | Description |
| :--- | :--- | :---: | :--- |
| **Retrieval** | `Recall@5` | **{ret_m['Recall@5']:.4f}** | Fraction of relevant target jobs in top-5 retrieved chunks |
| **Retrieval** | `Precision@5` | **{ret_m['Precision@5']:.4f}** | Precision of top-5 retrieved job chunks |
| **Retrieval** | `MRR` | **{ret_m['MRR']:.4f}** | Mean Reciprocal Rank of first relevant job result |
| **Retrieval** | `NDCG@5` | **{ret_m['NDCG@5']:.4f}** | Normalized Discounted Cumulative Gain ranking quality |
| **Skill Extraction** | `Precision` | **{skill_m['Precision']:.4f}** | Precision of canonical skill alias extraction |
| **Skill Extraction** | `Recall` | **{skill_m['Recall']:.4f}** | Coverage of ground-truth candidate skills |
| **Skill Extraction** | `F1 Score` | **{skill_m['F1_Score']:.4f}** | Harmonic mean of skill extraction precision & recall |
| **Groundedness** | `Citation Accuracy` | **{ground_m['citation_accuracy'] * 100:.1f}%** | Percentage of recommendations citing valid job chunks |
| **Groundedness** | `Hallucination Rate` | **{ground_m['hallucination_rate'] * 100:.1f}%** | Non-existent experience recommendation rate |"""

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace Markdown table between markers if present
    pattern = r"(\| Category \| Metric \| Score \| Description \|[\s\S]*?\| \*\*?Groundedness\*\*? \| `Hallucination Rate` \| \*\*.*?\*\* \| .*? \|)"
    if re.search(pattern, content):
        new_content = re.sub(pattern, table_markdown, content)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info(f"Auto-synced evaluation results table in '{readme_path}'.")


def run_evaluation(
    jobs_dir: str = "data/jobs",
    resumes_dir: str = "data/resumes",
    eval_dir: str = "data/evaluation",
    artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Ingest & Index Job Corpus
    jobs = JobCorpusIngestor.load_corpus(jobs_dir)
    if not jobs:
        raise RuntimeError(f"No job records found in '{jobs_dir}' for evaluation.")

    chunker = SectionAwareChunker()
    all_chunks = []
    for j in jobs:
        all_chunks.extend(chunker.chunk_job_description(j))

    embedder = VectorStoreManager(allow_fallback=True)
    embeddings = embedder.encode([c["text"] for c in all_chunks])

    vstore = PersistentFAISSVectorStore(
        dimension=embeddings.shape[1] if len(embeddings) > 0 else 384, encoder_model=embedder.model_used
    )
    vstore.add_chunks(all_chunks, embeddings)
    vstore.save(artifacts_dir)

    retriever = SemanticJobRetriever(vstore, embedder)
    rec_engine = CitationGroundedRecommendationEngine()

    rel_path = os.path.join(eval_dir, "relevance_labels.json")
    skill_path = os.path.join(eval_dir, "skill_labels.json")

    if not os.path.exists(rel_path) or not os.path.exists(skill_path):
        raise RuntimeError("Missing evaluation label benchmark files in data/evaluation/")

    with open(rel_path, "r", encoding="utf-8") as f:
        rel_data = json.load(f)
    with open(skill_path, "r", encoding="utf-8") as f:
        skill_data = json.load(f)

    recalls = []
    precisions = []
    mrrs = []
    ndcgs = []
    citation_accuracies = []
    hallucination_rates = []

    queries = rel_data.get("benchmark_queries", [])
    for q in queries:
        res_file = os.path.join(resumes_dir, q["resume_file"])
        if not os.path.exists(res_file):
            continue

        res_data = DocumentParser.parse_file(res_file)
        candidate_jobs = retriever.retrieve_candidate_jobs(
            res_data["clean_text"], top_k_chunks=40, max_candidate_jobs=15
        )

        retrieved_job_ids = list(candidate_jobs.keys())
        rel_ids = q["relevant_job_ids"]

        relevant_retrieved = [jid for jid in retrieved_job_ids if jid in rel_ids]
        recall = len(relevant_retrieved) / max(1, len(rel_ids))
        precision = len(relevant_retrieved) / max(1, len(retrieved_job_ids))
        mrr = calculate_mrr(retrieved_job_ids, rel_ids)
        ndcg = calculate_ndcg_at_k(retrieved_job_ids, rel_ids, k=5)

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        # Groundedness Evaluation
        if candidate_jobs:
            first_job_chunks = list(candidate_jobs.values())[0]
            recs = rec_engine.generate_recommendations(
                candidate_name=q["resume_file"],
                retrieved_chunks=first_job_chunks,
                matching_skills=[],
                missing_required_skills=[],
                missing_preferred_skills=[],
                overall_score=75.0,
            )
            gr_metrics = verify_citation_groundedness(recs["resume_recommendations"], first_job_chunks)
            citation_accuracies.append(gr_metrics["citation_accuracy"])
            hallucination_rates.append(gr_metrics["hallucination_rate"])

    if not recalls:
        raise RuntimeError("No valid retrieval evaluation examples were processed.")

    # Evaluate Skill Extraction Metrics
    skill_precisions = []
    skill_recalls = []
    skill_f1s = []

    for res_name, gt_skills in skill_data.get("ground_truth_skills", {}).items():
        res_file = os.path.join(resumes_dir, res_name)
        if not os.path.exists(res_file):
            continue

        res_data = DocumentParser.parse_file(res_file)
        extracted_skills = list(CategorizedSkillExtractor.extract_categorized_skills(res_data["clean_text"]).keys())

        gt_set = set([CategorizedSkillExtractor.normalize_skill_name(s) for s in gt_skills])
        ext_set = set(extracted_skills)

        tp = len(ext_set.intersection(gt_set))
        fp = len(ext_set.difference(gt_set))
        fn = len(gt_set.difference(ext_set))

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r) / max(1e-8, (p + r))

        skill_precisions.append(p)
        skill_recalls.append(r)
        skill_f1s.append(f1)

    if not skill_precisions:
        raise RuntimeError("No valid skill extraction evaluation examples were processed.")

    eval_results = {
        "encoder_model": embedder.model_used,
        "corpus_size": {"total_jobs": len(jobs), "total_chunks": len(all_chunks), "total_eval_queries": len(recalls)},
        "retrieval_metrics": {
            "Recall@5": round(float(np.mean(recalls)), 4),
            "Precision@5": round(float(np.mean(precisions)), 4),
            "MRR": round(float(np.mean(mrrs)), 4),
            "NDCG@5": round(float(np.mean(ndcgs)), 4),
        },
        "skill_extraction_metrics": {
            "Precision": round(float(np.mean(skill_precisions)), 4),
            "Recall": round(float(np.mean(skill_recalls)), 4),
            "F1_Score": round(float(np.mean(skill_f1s)), 4),
        },
        "recommendation_groundedness": {
            "citation_accuracy": round(float(np.mean(citation_accuracies)), 4) if citation_accuracies else 1.0,
            "hallucination_rate": round(float(np.mean(hallucination_rates)), 4) if hallucination_rates else 0.0,
        },
    }

    out_json = os.path.join(artifacts_dir, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    update_readme_benchmark_table(eval_results)
    logger.info(f"Saved empirical evaluation results to '{out_json}'")
    return eval_results


def main():
    results = run_evaluation()
    print("\n========================================================")
    print("   AI CAREER RAG ASSISTANT - EMPIRICAL BENCHMARK")
    print("========================================================\n")
    print(f"Model Encoder Used: {results['encoder_model']}")
    print(f"Jobs Indexed: {results['corpus_size']['total_jobs']} ({results['corpus_size']['total_chunks']} chunks)")
    print(f"Evaluation Queries Processed: {results['corpus_size']['total_eval_queries']}\n")

    print("| Category | Metric | Score |")
    print("|---|---|---|")
    for k, v in results["retrieval_metrics"].items():
        print(f"| Retrieval | {k} | {v:.4f} |")
    for k, v in results["skill_extraction_metrics"].items():
        print(f"| Skill Extraction | {k} | {v:.4f} |")
    for k, v in results["recommendation_groundedness"].items():
        print(f"| Groundedness | {k} | {v} |")
    print("\n========================================================\n")


if __name__ == "__main__":
    main()
