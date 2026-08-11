"""
AI Career RAG Assistant - Resume-Job Description Matching & Skill Gap Analysis
Author: Shivangi Srivastava (MS in AI @ NJIT)
"""

__version__ = "1.0.0"
__author__ = "Shivangi Srivastava"

from .embeddings import VectorStoreManager
from .matcher import CareerRAGMatcher
from .parser import DocumentParser
