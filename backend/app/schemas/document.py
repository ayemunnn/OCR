from datetime import datetime

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class DocumentResponse(BaseModel):
    id: int
    document_id: str
    original_filename: str
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    has_extracted_text: bool
    has_output_json: bool

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def add_file_flags(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return {
                "id": value.id,
                "document_id": value.document_id,
                "original_filename": value.original_filename,
                "status": value.status,
                "error_message": value.error_message,
                "created_at": value.created_at,
                "updated_at": value.updated_at,
                "has_extracted_text": bool(value.extracted_text_path),
                "has_output_json": bool(value.output_json_path),
            }

        value = value.copy()
        value["has_extracted_text"] = bool(value.get("extracted_text_path"))
        value["has_output_json"] = bool(value.get("output_json_path"))
        return value


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
