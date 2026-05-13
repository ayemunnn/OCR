from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.llm_service import analyze_document_text
from app.services.ocr_service import extract_text_from_images
from app.services.pdf_service import (
    cleanup_file,
    convert_pdf_bytes_to_images,
    save_temp_pdf,
    validate_pdf_filename,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/process")
async def process_document(file: UploadFile = File(...)):
    temp_path = None

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

        temp_path = save_temp_pdf(file_bytes)
        images = convert_pdf_bytes_to_images(file_bytes)
        extracted_text = extract_text_from_images(images)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
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
        except Exception as exc:
            llm_message = f"LLM processing failed: {exc}"
    else:
        llm_message = "LLM processing skipped because no OCR text was extracted."

    return {
        "filename": file.filename,
        "extracted_text_preview": extracted_text[:1000],
        "structured_output": structured_output,
        "status": llm_message,
    }
