import re
import os
from typing import List, Dict


class DocumentParser:
    """
    Parses resume and job description documents, performing text cleaning,
    section extraction, and semantic chunking.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans raw document text by stripping whitespace and removing special artifacts.
        """
        if not text:
            return ""
        # Remove extra whitespace and strange unicode symbols
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Extracts key resume sections (Skills, Education, Experience, Projects) using regex matching.
        """
        sections = {
            "skills": "",
            "experience": "",
            "education": "",
            "projects": "",
            "summary": ""
        }

        # Common headers
        patterns = {
            "skills": r'(?i)(skills|technical skills|technologies|expertise)(.*?)(experience|education|projects|work history|$)',
            "experience": r'(?i)(experience|work experience|employment|history)(.*?)(education|projects|skills|$)',
            "education": r'(?i)(education|academic background|degrees)(.*?)(experience|projects|skills|$)',
            "projects": r'(?i)(projects|key projects|portfolio)(.*?)(experience|education|skills|$)'
        }

        for sec, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                sections[sec] = DocumentParser.clean_text(match.group(2))

        if not any(sections.values()):
            sections["summary"] = DocumentParser.clean_text(text)

        return sections

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 256, chunk_overlap: int = 32) -> List[str]:
        """
        Splits long text into overlapping chunks for vector embedding index insertion.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - chunk_overlap)

        return chunks

    @classmethod
    def load_file(cls, filepath: str) -> str:
        """
        Loads document from TXT or PDF file path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.txt', '.md']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.pdf':
            try:
                import pypdf
                reader = pypdf.PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
