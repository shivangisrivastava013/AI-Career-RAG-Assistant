import pytest

from rag_assistant.parser import DocumentParser


def test_clean_text():
    raw = "Hello   world!\r\nThis\tis   a test."
    clean = DocumentParser.clean_text(raw)
    assert clean == "Hello world! This is a test."


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        DocumentParser.parse_file("non_existent_file.pdf")


def test_unsupported_format_raises_error(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("dummy text")
    with pytest.raises(ValueError):
        DocumentParser.parse_file(str(bad_file))


def test_txt_parsing(tmp_path):
    txt_file = tmp_path / "resume.txt"
    content = "EXPERIENCE\nSoftware Engineer at Tech Corp.\nSKILLS\nPython, PyTorch, SQL."
    txt_file.write_text(content)

    res = DocumentParser.parse_file(str(txt_file))
    assert res["file_name"] == "resume.txt"
    assert "experience" in res["sections"]
    assert "skills" in res["sections"]
    assert "Python" in res["clean_text"]


def test_empty_file_raises_error(tmp_path):
    empty_file = tmp_path / "empty_resume.txt"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        DocumentParser.parse_file(str(empty_file))


def test_whitespace_only_file_raises_error(tmp_path):
    ws_file = tmp_path / "whitespace_resume.txt"
    ws_file.write_text("   \n\t  \r\n   ")
    with pytest.raises(ValueError, match="no extractable text"):
        DocumentParser.parse_file(str(ws_file))


def test_malformed_pdf_raises_error(tmp_path):
    bad_pdf = tmp_path / "corrupted_resume.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4\ncorrupted random binary data \x00\xff\xfe\x01\x02")
    with pytest.raises(ValueError, match="Malformed or unreadable PDF"):
        DocumentParser.parse_file(str(bad_pdf))


def test_malformed_docx_raises_error(tmp_path):
    bad_docx = tmp_path / "corrupted_resume.docx"
    bad_docx.write_bytes(b"PK\x03\x04corrupted zip payload not a valid docx")
    with pytest.raises(ValueError, match="Malformed or unreadable DOCX"):
        DocumentParser.parse_file(str(bad_docx))


def test_load_file_method(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Senior Machine Learning Engineer with PyTorch experience.")
    text = DocumentParser.load_file(str(sample))
    assert "Senior Machine Learning Engineer" in text


def test_chunk_text_method():
    text = "Machine learning engineers build scalable inference pipelines using PyTorch and Docker."
    chunks = DocumentParser.chunk_text(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 0
    assert all(len(c) > 0 for c in chunks)
