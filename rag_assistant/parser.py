import re
import os
import logging
from typing import Dict, List, Any

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
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def parse_file(cls, file_path: str) -> Dict[str, Any]:
        """
        Parses document file by extension and returns structured text payload.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")

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

        sections = cls.extract_sections(raw_text)

        return {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "raw_text": raw_text,
            "clean_text": cls.clean_text(raw_text),
            "sections": sections
        }

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """
        Extracts plain text from PDF using pypdf.
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
            return "\n".join(pages_text)
        except Exception as e:
            logger.error(f"Error parsing PDF '{file_path}': {e}")
            raise RuntimeError(f"Could not extract text from PDF: {e}") from e

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """
        Extracts plain text from DOCX using python-docx.
        """
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(full_text)
        except Exception as e:
            logger.error(f"Error parsing DOCX '{file_path}': {e}")
            raise RuntimeError(f"Could not extract text from DOCX: {e}") from e

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
            "certifications": r"(?:certifications|licenses|credentials|courses)"
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
