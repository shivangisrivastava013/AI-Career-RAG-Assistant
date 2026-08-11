# AI Career RAG Assistant: Resume-Job Matching & Skill Gap Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-00599C?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent **Retrieval-Augmented Generation (RAG)** assistant that parses resumes and job descriptions, generates dense vector embeddings using Sentence-Transformers, indexes document sections, computes semantic similarity scores, identifies technical skill gaps, and delivers actionable career feedback.

---

## 🌟 Key Features
- 🧠 **Vector Embedding Retrieval:** Uses `sentence-transformers/all-MiniLM-L6-v2` to compute 384-dimensional dense semantic embeddings.
- 🎯 **Skill Gap Analysis:** Automatically extracts candidate skills vs required job competencies and flags missing key requirements.
- 📊 **Hybrid Matching Score:** Combines semantic vector cosine similarity (60% weight) with exact technical skill coverage (40% weight).
- 📑 **Multi-Format Document Parsing:** Supports parsing `.txt`, `.md`, and `.pdf` resume files.
- ⚡ **Instant Execution:** Run `demo.py` for automated evaluation.

---

## 🏗️ Architecture Flow

```text
[ Resume File (PDF/TXT) ] ------> [ Document Parser ] ------> [ Chunking & Cleaning ]
                                                                      |
                                                                      v
[ Job Description (TXT) ] -------> [ Vector Store Manager ] ---> [ Embedding Generation ]
                                                                      |
                                                                      v
                                                           [ RAG Cosine Similarity ]
                                                                      |
                                                                      v
                                                           [ Skill Gap & Feedback ]
```

---

## ⚡ Quick Start & Usage

```bash
# Clone the repository
git clone https://github.com/shivangisrivastava013/AI-Career-RAG-Assistant.git
cd AI-Career-RAG-Assistant

# Install dependencies
pip install -r requirements.txt

# Run demonstration
python demo.py
```

---

## 👤 Author
**Shivangi Srivastava**  
MS in Artificial Intelligence @ New Jersey Institute of Technology (NJIT)  
[LinkedIn Profile](https://www.linkedin.com/in/shivangisrivastava013/) | [Portfolio](https://shivangisrivastava013.github.io/shivangi-portfolio/)
