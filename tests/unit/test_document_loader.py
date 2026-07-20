from __future__ import annotations

import sys
import types
from pathlib import Path

from episcope.rag.ingestion.document_loader import GrobidDocumentLoader
from episcope.rag.ingestion.document_loader import UnstructuredDocumentLoader


def test_grobid_loader_uses_env_url(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, grobid_server: str) -> None:
            captured["grobid_server"] = grobid_server

    monkeypatch.setenv("GROBID_URL", "http://grobid:8070")
    monkeypatch.setitem(
        sys.modules,
        "episcope.rag.ingestion.local_grobid_client",
        types.SimpleNamespace(GrobidClient=FakeClient),
    )

    loader = GrobidDocumentLoader()

    assert loader.grobid_url == "http://grobid:8070"
    assert captured["grobid_server"] == "http://grobid:8070"


def test_grobid_loader_explicit_url_overrides_env(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, grobid_server: str) -> None:
            captured["grobid_server"] = grobid_server

    monkeypatch.setenv("GROBID_URL", "http://grobid:8070")
    monkeypatch.setitem(
        sys.modules,
        "episcope.rag.ingestion.local_grobid_client",
        types.SimpleNamespace(GrobidClient=FakeClient),
    )

    loader = GrobidDocumentLoader(grobid_url="http://custom-grobid:8070")

    assert loader.grobid_url == "http://custom-grobid:8070"
    assert captured["grobid_server"] == "http://custom-grobid:8070"


def test_unstructured_pdf_loader_falls_back_to_pdfminer(monkeypatch, tmp_path: Path) -> None:
    class FakeChar:
        def __init__(self, size: float, fontname: str) -> None:
            self.size = size
            self.fontname = fontname

    class FakeTextLine:
        def __init__(self, text: str, size: float = 10.0, bold: bool = False) -> None:
            self._text = text
            fontname = "Helvetica-Bold" if bold else "Helvetica"
            self._chars = [FakeChar(size, fontname) for _ in (text or " ")]

        def get_text(self) -> str:
            return self._text + "\n"

        def __iter__(self):
            return iter(self._chars)

    class FakeTextContainer:
        def __init__(self, lines) -> None:
            self._lines = lines

        def __iter__(self):
            return iter(self._lines)

    def fake_extract_pages(_path: str):
        yield [
            FakeTextContainer(
                [
                    FakeTextLine("Title page", size=14.0, bold=True),
                    FakeTextLine("Main finding one.", size=10.0),
                ]
            )
        ]
        yield [FakeTextContainer([FakeTextLine("Main finding two.", size=10.0)])]

    monkeypatch.setitem(
        sys.modules,
        "pdfminer.high_level",
        types.SimpleNamespace(extract_pages=fake_extract_pages),
    )
    monkeypatch.setitem(
        sys.modules,
        "pdfminer.layout",
        types.SimpleNamespace(
            LTChar=FakeChar, LTTextContainer=FakeTextContainer, LTTextLine=FakeTextLine
        ),
    )

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "unstructured.partition.pdf":
            raise ImportError("simulated missing unstructured PDF extras")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    sections, metadata, references = UnstructuredDocumentLoader().load(pdf_path)

    assert metadata.title == "paper"
    assert references == []
    assert sections
    combined_text = "\n".join(section.content for section in sections)
    assert "Main finding one." in combined_text
    assert "Main finding two." in combined_text
    assert any(section.title == "Title page" for section in sections)


def test_unstructured_pdf_loader_redirects_library_stdout_to_stderr(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    # `unstructured` prints diagnostics (e.g. "No languages specified...")
    # directly to stdout rather than logging, which corrupts `--format
    # json` CLI output. The loader must redirect that to stderr.
    class FakeElement:
        def __init__(self, text: str) -> None:
            self.text = text

    def fake_partition_pdf(**_kwargs):
        print("Warning: No languages specified, defaulting to English.")
        return [FakeElement("Title"), FakeElement("Body text.")]

    monkeypatch.setitem(
        sys.modules,
        "unstructured.partition.pdf",
        types.SimpleNamespace(partition_pdf=fake_partition_pdf),
    )
    monkeypatch.setitem(
        sys.modules,
        "unstructured.documents.elements",
        types.SimpleNamespace(Title=FakeElement, Header=FakeElement, Text=FakeElement),
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    UnstructuredDocumentLoader().load(pdf_path)

    captured = capsys.readouterr()
    assert "No languages specified" not in captured.out
    assert "No languages specified" in captured.err
