import re
from typing import List, Dict, Any


class SectionAwareChunker:
    """
    Section-aware document chunker that splits job descriptions and resumes into
    retrieval-optimized semantic chunks with section metadata.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_job_description(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits job description into section-aware chunks (Responsibilities, Qualifications, Preferred Skills).
        """
        job_id = job["job_id"]
        title = job["title"]
        company = job["company"]
        desc = job["description"]

        chunks = []
        chunk_idx = 0

        # Section pattern heuristic
        section_headers = r"(?:Responsibilities|Qualifications|Requirements|Preferred\s+Qualifications|About\s+the\s+Role|What\s+You'll\s+Do|Who\s+You\s+Are)"
        parts = re.split(f"({section_headers}:?)", desc, flags=re.IGNORECASE)

        if len(parts) > 1:
            current_section = "Overview"
            for i in range(1, len(parts), 2):
                header = parts[i].strip().rstrip(":")
                content = parts[i+1].strip() if i+1 < len(parts) else ""
                
                if content:
                    sub_chunks = self._sliding_window_chunks(content)
                    for sc in sub_chunks:
                        chunks.append({
                            "chunk_id": f"{job_id}_chunk_{chunk_idx}",
                            "job_id": job_id,
                            "title": title,
                            "company": company,
                            "section": header,
                            "text": f"{header}: {sc}"
                        })
                        chunk_idx += 1
        else:
            sub_chunks = self._sliding_window_chunks(desc)
            for sc in sub_chunks:
                chunks.append({
                    "chunk_id": f"{job_id}_chunk_{chunk_idx}",
                    "job_id": job_id,
                    "title": title,
                    "company": company,
                    "section": "General",
                    "text": sc
                })
                chunk_idx += 1

        return chunks

    def chunk_resume_sections(self, resume_parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits parsed resume sections into retrieval chunks.
        """
        sections = resume_parsed.get("sections", {})
        chunks = []
        chunk_idx = 0

        for sec_name, sec_text in sections.items():
            if not sec_text:
                continue

            sub_chunks = self._sliding_window_chunks(sec_text)
            for sc in sub_chunks:
                chunks.append({
                    "chunk_id": f"resume_chunk_{chunk_idx}",
                    "section": sec_name,
                    "text": sc
                })
                chunk_idx += 1

        return chunks

    def _sliding_window_chunks(self, text: str) -> List[str]:
        """
        Generates sliding window character chunks with word boundary preservation.
        """
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks = []
        current_words: List[str] = []
        current_len = 0

        for word in words:
            current_words.append(word)
            current_len += len(word) + 1

            if current_len >= self.chunk_size:
                chunks.append(" ".join(current_words))
                # Overlap words
                overlap_words = current_words[-max(1, self.chunk_overlap // 6):]
                current_words = list(overlap_words)
                current_len = sum(len(w) + 1 for w in current_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks
