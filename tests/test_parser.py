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
