import json
import uuid
from pathlib import Path
from typing import Any


STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage"


def ensure_storage_root(storage_root: Path = STORAGE_ROOT) -> Path:
    storage_root.mkdir(parents=True, exist_ok=True)
    return storage_root


def create_document_folder(storage_root: Path = STORAGE_ROOT) -> dict[str, Any]:
    root = ensure_storage_root(storage_root)
    document_id = uuid.uuid4().hex
    document_folder = root / document_id
    document_folder.mkdir(parents=True, exist_ok=False)

    return {
        "document_id": document_id,
        "folder_path": str(document_folder),
    }


def save_original_pdf(
    document_folder: str | Path,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    safe_filename = Path(filename).name or "uploaded.pdf"
    destination = Path(document_folder) / safe_filename
    destination.write_bytes(file_bytes)

    return {
        "original_pdf": {
            "filename": safe_filename,
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
        }
    }


def save_extracted_text(document_folder: str | Path, extracted_text: str) -> dict[str, Any]:
    destination = Path(document_folder) / "extracted_text.txt"
    destination.write_text(extracted_text, encoding="utf-8")

    return {
        "extracted_text": {
            "filename": destination.name,
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
        }
    }


def save_structured_output(
    document_folder: str | Path,
    structured_output: dict[str, Any] | None,
) -> dict[str, Any]:
    if not structured_output:
        return {}

    destination = Path(document_folder) / "output.json"
    destination.write_text(
        json.dumps(structured_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "structured_output": {
            "filename": destination.name,
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
        }
    }
