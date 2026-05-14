from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..models.document import Document
from ..services.llm_service import analyze_document_text
from ..services.ocr_service import extract_text_from_images
from ..services.pdf_service import (
    cleanup_file,
    convert_pdf_bytes_to_images,
    save_temp_pdf,
    validate_pdf_filename,
)
from ..services.storage_service import (
    create_document_folder,
    save_extracted_text,
    save_original_pdf,
    save_structured_output,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/process")
async def process_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    temp_path = None
    document_id = None
    document_folder = None
    saved_files = {}

    def saved_path(key: str) -> str | None:
        metadata = saved_files.get(key)
        if not metadata:
            return None
        return metadata.get("path")

    def save_document_record(status: str, error_message: str | None = None) -> None:
        if not document_id or not document_folder:
            return

        document = Document(
            document_id=document_id,
            original_filename=file.filename or "",
            storage_folder=document_folder,
            original_pdf_path=saved_path("original_pdf"),
            extracted_text_path=saved_path("extracted_text"),
            output_json_path=saved_path("structured_output"),
            status=status,
            error_message=error_message,
        )
        db.add(document)
        db.commit()

    try:
        validate_pdf_filename(file.filename)

        if file.content_type and file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a PDF file.",
            )

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

        document_metadata = create_document_folder()
        document_id = document_metadata["document_id"]
        document_folder = document_metadata["folder_path"]
        saved_files.update(
            save_original_pdf(document_folder, file.filename, file_bytes)
        )

        temp_path = save_temp_pdf(file_bytes)
        images = convert_pdf_bytes_to_images(file_bytes)
        extracted_text = extract_text_from_images(images)
        saved_files.update(save_extracted_text(document_folder, extracted_text))
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        save_document_record("failed", str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {exc}",
        ) from exc
    finally:
        cleanup_file(temp_path)

    structured_output = None
    llm_message = None
    if extracted_text:
        try:
            llm_result = analyze_document_text(extracted_text)
            structured_output = llm_result["structured_output"]
            llm_message = llm_result["message"]
            saved_files.update(
                save_structured_output(document_folder, structured_output)
            )
        except Exception as exc:
            llm_message = f"LLM processing failed: {exc}"
    else:
        llm_message = "LLM processing skipped because no OCR text was extracted."

    save_document_record(llm_message)

    return {
        "filename": file.filename,
        "status": llm_message,
        "document_id": document_id,
        "extracted_text_preview": extracted_text[:1000],
        "structured_output": structured_output,
        "saved_files": saved_files,
    }
