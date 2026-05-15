import os
import tempfile
from pathlib import Path

from pdf2image import convert_from_bytes


def validate_pdf_filename(filename: str | None) -> None:
    if not filename or not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF uploads are supported by this endpoint.")


def save_temp_pdf(file_bytes: bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        temp_file.write(file_bytes)
        return temp_file.name
    finally:
        temp_file.close()


def convert_pdf_bytes_to_images(file_bytes: bytes):
    return convert_from_bytes(file_bytes)


def cleanup_file(path: str | Path | None) -> None:
    if not path:
        return

    try:
        os.remove(path)
    except FileNotFoundError:
        return
