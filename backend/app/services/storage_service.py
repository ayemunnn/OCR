import json
import uuid
from pathlib import Path
from typing import Any

from ..core.config import get_settings
from .azure_storage_service import AzureStorageConfigurationError


STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage"


class UnsupportedStorageProviderError(RuntimeError):
    pass


def _get_storage_provider() -> str:
    provider = get_settings().storage_provider.lower().strip()
    if provider not in {"local", "azure"}:
        raise UnsupportedStorageProviderError(
            f"Unsupported storage provider '{provider}'. Use 'local' or 'azure'."
        )
    return provider


def ensure_storage_root(storage_root: Path = STORAGE_ROOT) -> Path:
    storage_root.mkdir(parents=True, exist_ok=True)
    return storage_root


def create_document_folder(storage_root: Path = STORAGE_ROOT) -> dict[str, Any]:
    document_id = uuid.uuid4().hex

    if _get_storage_provider() == "azure":
        return {
            "document_id": document_id,
            "folder_path": document_id,
        }

    root = ensure_storage_root(storage_root)
    document_folder = root / document_id
    document_folder.mkdir(parents=True, exist_ok=False)

    return {
        "document_id": document_id,
        "folder_path": str(document_folder),
    }


def save_upload(
    document_folder: str | Path,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    return save_original_pdf(document_folder, filename, file_bytes)


def save_original_pdf(
    document_folder: str | Path,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    if _get_storage_provider() == "azure":
        from . import azure_storage_service

        return azure_storage_service.save_original_pdf(
            str(document_folder),
            filename,
            file_bytes,
        )

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
    if _get_storage_provider() == "azure":
        from . import azure_storage_service

        return azure_storage_service.save_extracted_text(
            str(document_folder),
            extracted_text,
        )

    destination = Path(document_folder) / "extracted_text.txt"
    destination.write_text(extracted_text, encoding="utf-8")

    return {
        "extracted_text": {
            "filename": destination.name,
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
        }
    }


def save_output_json(
    document_folder: str | Path,
    structured_output: dict[str, Any] | None,
) -> dict[str, Any]:
    return save_structured_output(document_folder, structured_output)


def save_structured_output(
    document_folder: str | Path,
    structured_output: dict[str, Any] | None,
) -> dict[str, Any]:
    if not structured_output:
        return {}

    if _get_storage_provider() == "azure":
        from . import azure_storage_service

        return azure_storage_service.save_output_json(
            str(document_folder),
            structured_output,
        )

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


def read_extracted_text(path: str | Path | None) -> str:
    if _get_storage_provider() == "azure":
        from . import azure_storage_service

        return azure_storage_service.read_extracted_text(str(path) if path else None)

    if not path:
        raise FileNotFoundError("Extracted text file is not available.")

    text_path = Path(path)
    if not text_path.exists():
        raise FileNotFoundError("Extracted text file is not available.")

    return text_path.read_text(encoding="utf-8")


def read_output_json(path: str | Path | None) -> dict[str, Any]:
    if _get_storage_provider() == "azure":
        from . import azure_storage_service

        return azure_storage_service.read_output_json(str(path) if path else None)

    if not path:
        raise FileNotFoundError("Output JSON file is not available.")

    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError("Output JSON file is not available.")

    return json.loads(json_path.read_text(encoding="utf-8"))
