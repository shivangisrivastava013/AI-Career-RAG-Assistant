import os
from rag_assistant.matcher import CareerRAGMatcher
from rag_assistant.parser import DocumentParser


def main():
    print("[+] Initializing AI Career RAG Assistant...")
    
    resume_path = "./data/sample_resume.txt"
    job_path = "./data/sample_job.txt"

    if not os.path.exists(resume_path) or not os.path.exists(job_path):
        print("[!] Sample files missing in ./data/")
        return

    resume_text = DocumentParser.load_file(resume_path)
    job_text = DocumentParser.load_file(job_path)

    matcher = CareerRAGMatcher()
    results = matcher.evaluate_match(resume_text, job_text)

    print("\n" + "="*60)
    print("[*] RAG EVALUATION & RESUME MATCH RESULTS")
    print("="*60)
    print(f"[*] Overall Match Score:         {results['overall_match_percentage']}%")
    print(f"[*] Semantic Vector Similarity: {results['semantic_similarity_score']}")
    print(f"[*] Skill Coverage Ratio:        {results['skill_coverage_ratio'] * 100:.1f}%")
    print(f"[+] Matching Skills Found:       {', '.join(results['matching_skills'])}")
    print(f"[!] Missing Skills Identified:    {', '.join(results['missing_skills']) if results['missing_skills'] else 'None'}")
    print("\n[Feedback] Recommendation:")
    print(f"   {results['recommendation']}")
    print("="*60)


if __name__ == '__main__':
    main()
