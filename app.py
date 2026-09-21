import json
import os
import tempfile

import streamlit as st

from rag_assistant.chunking import SectionAwareChunker
from rag_assistant.embeddings import VectorStoreManager
from rag_assistant.ingestion import JobCorpusIngestor
from rag_assistant.parser import DocumentParser
from rag_assistant.ranking import ConfigurableJobRanker
from rag_assistant.recommendation import CitationGroundedRecommendationEngine
from rag_assistant.retrieval import SemanticJobRetriever
from rag_assistant.skill_extraction import CategorizedSkillExtractor
from rag_assistant.vector_store import PersistentFAISSVectorStore

st.set_page_config(
    page_title="AI Career RAG Assistant", page_icon="🎯", layout="wide", initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #06b6d4; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #94a3b8; margin-bottom: 20px; }
    .card-box { background-color: #0f172a; padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px; }
    .score-badge { background-color: rgba(6, 182, 212, 0.15); color: #06b6d4; padding: 6px 14px; border-radius: 4px; font-weight: bold; border: 1px solid rgba(6, 182, 212, 0.3); }
    .skill-chip { display: inline-block; background-color: rgba(16, 185, 129, 0.15); color: #10b981; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; margin-right: 6px; margin-bottom: 6px; border: 1px solid rgba(16, 185, 129, 0.3); }
    .missing-chip { display: inline-block; background-color: rgba(239, 68, 68, 0.15); color: #ef4444; padding: 3px 10px; border-radius: 4px; font-size: 0.85rem; margin-right: 6px; margin-bottom: 6px; border: 1px solid rgba(239, 68, 68, 0.3); }
</style>
""",
    unsafe_allow_html=True,
)


def load_artifacts(artifacts_dir="artifacts", allow_fallback=True):
    vstore = PersistentFAISSVectorStore()
    if os.path.exists(os.path.join(artifacts_dir, "job_metadata.json")):
        vstore.load(artifacts_dir, expected_encoder=None)
    embedder = VectorStoreManager(allow_fallback=allow_fallback)
    return vstore, embedder


def main():
    st.markdown('<div class="main-header">🎯 AI Career RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Portfolio-grade Retrieval-Augmented Career Matcher & Grounded Skill Alignment Engine</div>',
        unsafe_allow_html=True,
    )

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("data/jobs", exist_ok=True)

    # Sidebar Config
    st.sidebar.header("⚙️ System Configuration")
    top_k_chunks = st.sidebar.slider("Top-K Retrieved Candidate Chunks", min_value=10, max_value=60, value=40)
    top_k_jobs = st.sidebar.slider("Final Ranked Jobs to Display", min_value=1, max_value=10, value=5)
    allow_fallback = st.sidebar.checkbox(
        "Allow SHA-256 Fallback",
        value=True,
        help="Explicitly allow SHA-256 feature encoding if transformer model fails to load.",
    )
    config_file = st.sidebar.text_input("Scoring Config Path", value="config/scoring.yaml")

    vstore, embedder = load_artifacts(artifacts_dir, allow_fallback=allow_fallback)
    st.sidebar.info(f"**Model In Use:** `{embedder.model_used}`\n\n**Indexed Chunks:** `{len(vstore.metadata_store)}`")

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📄 Resume Matcher",
            "📂 Job Corpus Indexer",
            "📊 Skill Gap Matrix",
            "💡 Grounded Recommendations",
            "📥 Export Report",
        ]
    )

    # TAB 1: RESUME MATCHER
    with tab1:
        st.subheader("1. Candidate Resume Analysis & Job Search")
        uploaded_file = st.file_uploader("Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

        resume_text = ""
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                parsed = DocumentParser.parse_file(tmp_path)
                resume_text = parsed["clean_text"]
                st.success(f"Successfully parsed '{uploaded_file.name}' ({len(resume_text.split())} words)")
                with st.expander("Preview Parsed Resume Sections"):
                    st.json(parsed["sections"])
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            sample_txt_path = "data/resumes/sample_resume.txt"
            if os.path.exists(sample_txt_path):
                with open(sample_txt_path, "r", encoding="utf-8") as f:
                    resume_text = f.read()
                st.info(
                    "Using default sample resume (`data/resumes/sample_resume.txt`). Upload your own resume above to test!"
                )

        if resume_text:
            if len(vstore.metadata_store) == 0:
                st.warning(
                    "No jobs indexed in FAISS vector store yet. Please go to Tab 2 ('Job Corpus Indexer') to build the index!"
                )
            else:
                retriever = SemanticJobRetriever(vstore, embedder)
                candidate_jobs = retriever.retrieve_candidate_jobs(
                    resume_text, top_k_chunks=top_k_chunks, max_candidate_jobs=15
                )
                ranker = ConfigurableJobRanker(config_path=config_file)
                rec_engine = CitationGroundedRecommendationEngine()

                eval_results = []
                for job_id, j_chunks in candidate_jobs.items():
                    first_c = j_chunks[0]
                    j_title = first_c.get("title", "Target Position")
                    j_company = first_c.get("company", "Company")
                    j_req_skills = list(first_c.get("required_skills", []))
                    j_pref_skills = list(first_c.get("preferred_skills", []))
                    j_text = " ".join([c["text"] for c in j_chunks])

                    skill_comp = CategorizedSkillExtractor.compare_skills(
                        resume_text=resume_text,
                        job_required_skills=j_req_skills,
                        job_preferred_skills=j_pref_skills,
                        job_description_text=j_text,
                    )

                    chunk_sim = max([c["similarity_score"] for c in j_chunks], default=0.0)
                    metrics = ranker.compute_composite_score(
                        semantic_responsibility_sim=chunk_sim,
                        required_skill_coverage=skill_comp["required_coverage"],
                        preferred_skill_coverage=skill_comp["preferred_coverage"] if j_pref_skills else None,
                        resume_text=resume_text,
                        job={"title": j_title, "description": j_text},
                    )

                    recs = rec_engine.generate_recommendations(
                        candidate_name="Candidate",
                        retrieved_chunks=j_chunks,
                        matching_skills=skill_comp["matching_skills"],
                        missing_required_skills=skill_comp["missing_required_skills"],
                        missing_preferred_skills=skill_comp["missing_preferred_skills"],
                        overall_score=metrics["overall_match_score"],
                    )

                    eval_results.append(
                        {
                            "job_id": job_id,
                            "title": j_title,
                            "company": j_company,
                            "metrics": metrics,
                            "skill_comp": skill_comp,
                            "recs": recs,
                            "chunks": j_chunks,
                        }
                    )

                eval_results.sort(key=lambda x: x["metrics"]["overall_match_score"], reverse=True)
                final_results = eval_results[:top_k_jobs]
                st.session_state["eval_results"] = final_results

                st.subheader(
                    f"Ranked Compatibility Results ({len(final_results)} Top Jobs Selected from {len(eval_results)} Candidates)"
                )
                for idx, item in enumerate(final_results, 1):
                    score = item["metrics"]["overall_match_score"]
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### #{idx} {item['title']} @ **{item['company']}**")
                            st.markdown(f"**Job ID:** `{item['job_id']}`")
                        with col2:
                            st.metric("Compatibility Score", f"{score}%")

                        st.write("**Component Breakdown:**")
                        bd = item["metrics"]["component_breakdown"]
                        b_cols = st.columns(5)
                        b_cols[0].metric(
                            "Required Skills",
                            (
                                f"{bd['required_skills']}%"
                                if isinstance(bd["required_skills"], (int, float))
                                else bd["required_skills"]
                            ),
                        )
                        b_cols[1].metric(
                            "Responsibilities",
                            (
                                f"{bd['responsibilities']}%"
                                if isinstance(bd["responsibilities"], (int, float))
                                else bd["responsibilities"]
                            ),
                        )
                        b_cols[2].metric(
                            "Experience",
                            (
                                f"{bd['experience_level']}%"
                                if isinstance(bd["experience_level"], (int, float))
                                else bd["experience_level"]
                            ),
                        )
                        b_cols[3].metric(
                            "Education",
                            (
                                f"{bd['education_level']}%"
                                if isinstance(bd["education_level"], (int, float))
                                else bd["education_level"]
                            ),
                        )
                        b_cols[4].metric(
                            "Preferred",
                            (
                                f"{bd['preferred_skills']}%"
                                if isinstance(bd["preferred_skills"], (int, float))
                                else bd["preferred_skills"]
                            ),
                        )

                        m_skills = [m["skill"] for m in item["skill_comp"]["matching_skills"]]
                        g_skills = [g["skill"] for g in item["skill_comp"]["missing_required_skills"]]

                        st.write(f"**Matching Skills ({len(m_skills)}):** " + ", ".join(m_skills[:8]))
                        if g_skills:
                            st.write(f"**Missing Skills ({len(g_skills)}):** " + ", ".join(g_skills[:8]))

                        st.divider()

    # TAB 2: JOB CORPUS INDEXER
    with tab2:
        st.subheader("2. Index Synthetic Job Corpus into FAISS Vector Database")
        st.write(
            "Ingest job description JSON records, extract section chunks with structured skill metadata, and persist FAISS index (`artifacts/jobs.faiss`)."
        )

        jobs_dir = st.text_input("Job Directory Path", value="data/jobs")
        if st.button("🚀 Rebuild FAISS Vector Index Now"):
            with st.spinner("Chunking & Embedding Job Corpus..."):
                jobs = JobCorpusIngestor.load_corpus(jobs_dir)
                if not jobs:
                    st.error(f"No job JSON files found in {jobs_dir}")
                else:
                    chunker = SectionAwareChunker()
                    all_chunks = []
                    for j in jobs:
                        all_chunks.extend(chunker.chunk_job_description(j))

                    texts = [c["text"] for c in all_chunks]
                    embeddings = embedder.encode(texts)

                    new_vstore = PersistentFAISSVectorStore(
                        dimension=embeddings.shape[1] if len(embeddings) > 0 else 384, encoder_model=embedder.model_used
                    )
                    new_vstore.add_chunks(all_chunks, embeddings)
                    new_vstore.save(artifacts_dir)

                    st.success(
                        f"Successfully indexed {len(jobs)} synthetic jobs ({len(all_chunks)} chunks) into FAISS!"
                    )
                    st.rerun()

    # TAB 3: SKILL GAP MATRIX
    with tab3:
        st.subheader("3. Categorized Skill Gap & Evidence Viewer")
        if "eval_results" in st.session_state and st.session_state["eval_results"]:
            res_list = st.session_state["eval_results"]
            job_titles = [f"{r['title']} @ {r['company']}" for r in res_list]
            sel_idx = st.selectbox(
                "Select Target Job Role", range(len(job_titles)), format_func=lambda i: job_titles[i]
            )
            sel_res = res_list[sel_idx]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### ✅ Matching Candidate Skills")
                for match in sel_res["skill_comp"]["matching_skills"]:
                    st.markdown(f"**• {match['skill']}** (`{match['category']}`)")
                    st.caption(f"**Resume Evidence:** \"{match['resume_evidence']}\"")
                    st.caption(f"**Job Context:** \"{match['job_evidence']}\"")
                    st.write("---")

            with col_b:
                st.markdown("#### ❌ Missing Skill Gaps")
                for gap in sel_res["skill_comp"]["missing_required_skills"]:
                    st.markdown(f"**• {gap['skill']}** (`{gap['category']}`)")
                    st.caption(f"**Job Context:** \"{gap['job_evidence']}\"")
                    st.write("---")
        else:
            st.info("Run resume analysis on Tab 1 first to view categorized skill gap details.")

    # TAB 4: GROUNDED RECOMMENDATIONS
    with tab4:
        st.subheader("4. Citation-Grounded Recommendations")
        if "eval_results" in st.session_state and st.session_state["eval_results"]:
            res_list = st.session_state["eval_results"]
            sel_idx_rec = st.selectbox(
                "Select Target Role for Recommendations",
                range(len(res_list)),
                format_func=lambda i: f"{res_list[i]['title']} @ {res_list[i]['company']}",
            )
            target_item = res_list[sel_idx_rec]
            recs = target_item["recs"]

            st.info(recs["fit_summary"])

            st.markdown("### 📌 Actionable Resume Recommendations")
            for rec in recs["resume_recommendations"]:
                st.markdown(f"**[{rec['type']}] {rec['action']}**")
                st.write(rec["detail"])
                st.caption(f"📖 **Cited Job Requirement Chunk:** `{rec['cited_job_chunk']}`")
                st.write("---")

            st.markdown("### 💬 Recommended Interview Preparation Topics")
            for topic in recs["interview_topics"]:
                st.markdown(f"• {topic}")
        else:
            st.info("Run resume analysis on Tab 1 first to generate grounded recommendations.")

    # TAB 5: EXPORT REPORT
    with tab5:
        st.subheader("5. Export Analysis Report")
        if "eval_results" in st.session_state and st.session_state["eval_results"]:
            res_list = st.session_state["eval_results"]
            export_payload = {
                "system_model": embedder.model_used,
                "total_jobs_evaluated": len(res_list),
                "top_match": res_list[0] if res_list else {},
            }
            json_str = json.dumps(export_payload, indent=2)
            st.download_button(
                "📥 Download Analysis Report (JSON)",
                data=json_str,
                file_name="rag_career_analysis.json",
                mime="application/json",
            )
        else:
            st.info("Run resume analysis on Tab 1 first to export reports.")


if __name__ == "__main__":
    main()
