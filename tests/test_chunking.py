from rag_assistant.chunking import SectionAwareChunker


def test_chunk_job_description():
    job = {
        "job_id": "job_100",
        "title": "AI Engineer",
        "company": "Test Company",
        "description": "Responsibilities: Develop deep learning models using PyTorch. Qualifications: 2+ years PyTorch experience. Docker experience.",
    }

    chunker = SectionAwareChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_job_description(job)

    assert len(chunks) > 0
    assert chunks[0]["job_id"] == "job_100"
    assert chunks[0]["title"] == "AI Engineer"
    assert "text" in chunks[0]


def test_empty_description_chunking():
    job = {"job_id": "job_101", "title": "Empty Role", "company": "Test Company", "description": ""}
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_job_description(job)
    assert chunks == []
