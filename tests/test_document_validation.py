"""Tests for upload validation in the document processor (P0 hardening).

These exercise magic-byte sniffing and the size cap, which guard against
oversized uploads and extension/MIME spoofing. They import only the
processor module (stdlib-only at import time) so they run without the
heavy ML/OCR stack installed.
"""
import pytest

from ai_court.documents.processor import (
    DocumentProcessor,
    _MAX_DOCUMENT_BYTES,
    _sniff_kind,
    _validate_upload,
)


def test_sniff_kind_recognizes_signatures():
    assert _sniff_kind(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3") == "pdf"
    assert _sniff_kind(b"\x89PNG\r\n\x1a\nrest-of-png") == "image"
    assert _sniff_kind(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "image"
    assert _sniff_kind(b"GIF89a\x01\x00") == "image"
    assert _sniff_kind(b"BM\x36\x00") == "image"
    assert _sniff_kind(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image"
    assert _sniff_kind(b"PK\x03\x04docx-zip-body") == "docx"
    assert _sniff_kind(b"Plain legal text about IPC 302.") == "text"
    assert _sniff_kind(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc") == "unknown"
    assert _sniff_kind(b"") == "unknown"


def test_validate_upload_rejects_empty():
    with pytest.raises(ValueError):
        _validate_upload(b"", "empty.txt")


def test_validate_upload_rejects_oversized():
    oversized = b"a" * (_MAX_DOCUMENT_BYTES + 1)
    with pytest.raises(ValueError):
        _validate_upload(oversized, "huge.txt")


def test_validate_upload_rejects_unknown_binary():
    with pytest.raises(ValueError):
        _validate_upload(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc\xfb", "mystery.bin")


def test_validate_upload_returns_kind_for_text():
    assert _validate_upload(b"Section 420 IPC cheating case.", "note.txt") == "text"


def test_process_file_rejects_oversized():
    proc = DocumentProcessor()
    oversized = b"a" * (_MAX_DOCUMENT_BYTES + 1)
    with pytest.raises(ValueError):
        proc.process_file(file_bytes=oversized, filename="huge.txt")


def test_process_file_routes_by_content_not_extension():
    # File claims to be a PDF but is plain text: must be handled as text,
    # never handed to the PDF parser (defends against extension spoofing).
    proc = DocumentProcessor()
    doc = proc.process_file(
        file_bytes=b"Plain text disguised as a PDF. Section 302 IPC.",
        filename="evidence.pdf",
    )
    assert doc.file_type == "txt"
    assert "Section 302" in doc.text


def test_process_file_rejects_disguised_binary():
    proc = DocumentProcessor()
    with pytest.raises(ValueError):
        proc.process_file(
            file_bytes=b"\x00\x01\x02\x03\xff\xfe\xfd",
            filename="evil.pdf",
        )
