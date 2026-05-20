import json
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient, ContentSettings

from ..core.config import get_settings


class AzureStorageConfigurationError(RuntimeError):
    pass


def _get_container_client():
    settings = get_settings()
    if not settings.azure_storage_connection_string:
        raise AzureStorageConfigurationError(
            "Azure storage is selected but AZURE_STORAGE_CONNECTION_STRING is not configured."
        )

    blob_service_client = BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )
    container_client = blob_service_client.get_container_client(
        settings.azure_storage_container_name
    )
    if not container_client.exists():
        container_client.create_container()
    return container_client


def _blob_name(document_folder: str, filename: str) -> str:
    safe_filename = Path(filename).name
    return f"{document_folder}/{safe_filename}"


def save_original_pdf(
    document_folder: str,
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    safe_filename = Path(filename).name or "uploaded.pdf"
    blob_name = _blob_name(document_folder, safe_filename)
    container_client = _get_container_client()
    container_client.upload_blob(
        name=blob_name,
        data=file_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/pdf"),
    )

    return {
        "original_pdf": {
            "filename": safe_filename,
            "path": blob_name,
            "size_bytes": len(file_bytes),
        }
    }


def save_extracted_text(document_folder: str, extracted_text: str) -> dict[str, Any]:
    blob_name = _blob_name(document_folder, "extracted_text.txt")
    text_bytes = extracted_text.encode("utf-8")
    container_client = _get_container_client()
    container_client.upload_blob(
        name=blob_name,
        data=text_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type="text/plain; charset=utf-8"),
    )

    return {
        "extracted_text": {
            "filename": "extracted_text.txt",
            "path": blob_name,
            "size_bytes": len(text_bytes),
        }
    }


def save_output_json(
    document_folder: str,
    structured_output: dict[str, Any] | None,
) -> dict[str, Any]:
    if not structured_output:
        return {}

    blob_name = _blob_name(document_folder, "output.json")
    output = json.dumps(structured_output, indent=2, ensure_ascii=False)
    output_bytes = output.encode("utf-8")
    container_client = _get_container_client()
    container_client.upload_blob(
        name=blob_name,
        data=output_bytes,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )

    return {
        "structured_output": {
            "filename": "output.json",
            "path": blob_name,
            "size_bytes": len(output_bytes),
        }
    }


def read_extracted_text(blob_name: str | None) -> str:
    if not blob_name:
        raise FileNotFoundError("Extracted text file is not available.")

    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    if not blob_client.exists():
        raise FileNotFoundError("Extracted text file is not available.")

    return blob_client.download_blob().readall().decode("utf-8")


def read_output_json(blob_name: str | None) -> dict[str, Any]:
    if not blob_name:
        raise FileNotFoundError("Output JSON file is not available.")

    container_client = _get_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    if not blob_client.exists():
        raise FileNotFoundError("Output JSON file is not available.")

    raw_output = blob_client.download_blob().readall().decode("utf-8")
    return json.loads(raw_output)
