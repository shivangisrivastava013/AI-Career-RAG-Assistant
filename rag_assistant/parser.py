import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Multi-format resume and job description parser supporting PDF, DOCX, and TXT files
    with section identification and text normalization.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalizes line endings, whitespace, and removes non-printable characters.
        """
        if not text:
            return ""
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def load_file(cls, file_path: str) -> str:
        """
        Loads document file and returns cleaned plain text.
        Raises ValueError if empty or malformed.
        """
        parsed = cls.parse_file(file_path)
        return parsed["clean_text"]

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 128, chunk_overlap: int = 16) -> List[str]:
        """
        Splits clean text into sliding window chunks.
        """
        if not text or not text.strip():
            return []
        words = text.split()
        if not words:
            return []
        chunks = []
        current: List[str] = []
        curr_len = 0
        for w in words:
            current.append(w)
            curr_len += len(w) + 1
            if curr_len >= chunk_size:
                chunks.append(" ".join(current))
                overlap_count = max(1, chunk_overlap // 8)
                current = current[-overlap_count:]
                curr_len = sum(len(x) + 1 for x in current)
        if current:
            chunks.append(" ".join(current))
        return chunks

    @classmethod
    def parse_file(cls, file_path: str) -> Dict[str, Any]:
        """
        Parses document file by extension and returns structured text payload.
        Raises ValueError for empty files, unreadable content, or malformed formats.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"Uploaded resume document '{os.path.basename(file_path)}' is empty (0 bytes).")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            raw_text = cls._parse_pdf(file_path)
        elif ext == ".docx":
            raw_text = cls._parse_docx(file_path)
        elif ext in [".txt", ".json", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        else:
            raise ValueError(f"Unsupported document format '{ext}'. Supported formats: .pdf, .docx, .txt")

        clean_text = cls.clean_text(raw_text)
        if not clean_text:
            raise ValueError(
                f"Uploaded resume document '{os.path.basename(file_path)}' contains no extractable text."
            )

        sections = cls.extract_sections(raw_text)

        return {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "sections": sections,
        }

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """
        Extracts plain text from PDF using pypdf.
        Raises ValueError if PDF is corrupted or unreadable.
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
            extracted = "\n".join(pages_text).strip()
            if not extracted:
                raise ValueError(
                    f"Uploaded PDF document '{os.path.basename(file_path)}' contains no extractable text."
                )
            return extracted
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error parsing PDF '{file_path}': {e}")
            raise ValueError(f"Malformed or unreadable PDF document '{os.path.basename(file_path)}': {e}") from e

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """
        Extracts plain text from DOCX using python-docx.
        Raises ValueError if DOCX is corrupted or unreadable.
        """
        try:
            import docx

            doc = docx.Document(file_path)
            full_text = [p.text for p in doc.paragraphs if p.text.strip()]
            extracted = "\n".join(full_text).strip()
            if not extracted:
                raise ValueError(
                    f"Uploaded DOCX document '{os.path.basename(file_path)}' contains no extractable text."
                )
            return extracted
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error parsing DOCX '{file_path}': {e}")
            raise ValueError(f"Malformed or unreadable DOCX document '{os.path.basename(file_path)}': {e}") from e

    @classmethod
    def extract_sections(cls, text: str) -> Dict[str, str]:
        """
        Parses raw text into standard resume / JD sections using structural heading regex.
        """
        section_headers = {
            "summary": r"(?:summary|objective|profile|about\s+me)",
            "experience": r"(?:experience|work\s+history|employment|professional\s+experience)",
            "education": r"(?:education|academic\s+background|degrees)",
            "skills": r"(?:skills|technical\s+skills|core\s+competencies|technologies)",
            "projects": r"(?:projects|key\s+projects|featured\s+work)",
            "certifications": r"(?:certifications|licenses|credentials|courses)",
        }

        sections: Dict[str, str] = {sec: "" for sec in section_headers}
        sections["general"] = ""

        # Split text by line blocks
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        current_section = "general"
        buf: Dict[str, List[str]] = {sec: [] for sec in sections}

        for line in lines:
            line_lower = line.lower()
            matched_sec = None
            for sec, pattern in section_headers.items():
                if re.match(r"^#*\s*" + pattern + r"\b", line_lower):
                    matched_sec = sec
                    break

            if matched_sec:
                current_section = matched_sec
            else:
                buf[current_section].append(line)

        for sec in sections:
            sections[sec] = " ".join(buf[sec]).strip()

        return sections
