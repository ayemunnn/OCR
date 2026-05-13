# PaperSleuth OCR Refactor Plan

## Current Streamlit App Summary

The current `paperSleuth.py` app is a single-file Streamlit application that lets a user upload a PDF or image, previews the uploaded content, extracts text with Tesseract OCR, sends the OCR text to a Hugging Face-hosted LLM for document skimming and structured extraction, displays the returned JSON, and generates a downloadable PDF summary.

The app loads environment variables with `python-dotenv`, expects `HF_API_KEY` to be present, uses a fixed Hugging Face model name, and creates a global `InferenceClient` before rendering the Streamlit UI.

## Current Functional Sections

- Streamlit UI code:
  - Page setup with `st.set_page_config`.
  - Title, caption, uploader, preview images, text area, analyze button, status messages, JSON display, and download button.

- PDF upload handling:
  - `st.file_uploader` accepts PDF, PNG, JPG, and JPEG files.
  - Uploaded file type is read from `uploaded_file.type`.
  - PDF bytes are read with `uploaded_file.read()`.

- PDF-to-image conversion:
  - PDFs are converted with `pdf2image.convert_from_bytes`.
  - Each converted page image is displayed in Streamlit before OCR.

- OCR/text extraction:
  - PDF page images are processed with `pytesseract.image_to_string`.
  - Uploaded image files are opened with `PIL.Image.open` and processed with `pytesseract.image_to_string`.
  - OCR text from multiple PDF pages is joined with blank lines.
  - The user can review and edit OCR text in a Streamlit text area before LLM analysis.

- LLM cleanup/extraction logic:
  - A hardcoded `MODEL_NAME` of `google/gemma-3-27b-it` is used.
  - `InferenceClient` is initialized with the Hugging Face API key from `HF_API_KEY`.
  - A detailed system prompt asks the model to return only valid JSON with `summary`, `key_points`, `entities`, and `metadata`.
  - The app calls `client.chat_completion` with a low temperature and max token limit.

- JSON output generation:
  - The app extracts the model response content from the first completion choice.
  - It strips possible markdown code fences.
  - It attempts to isolate JSON between the first `{` and last `}`.
  - It parses the candidate JSON with `json.loads`.

- Download/display logic:
  - Parsed data is displayed with `st.json`.
  - A PDF summary is generated with `FPDF`.
  - The summary includes the parsed summary, key points, entities, and metadata.
  - The generated PDF is written to `document_skimmer_summary.pdf` and exposed through `st.download_button`.

- Hardcoded secrets, tokens, or unsafe config:
  - No API key is hardcoded in the script; `HF_API_KEY` is read from the environment.
  - The Hugging Face model name is hardcoded as app configuration.
  - The app raises a `ValueError` at import time if `HF_API_KEY` is missing.
  - The generated PDF uses a fixed filename in the working directory.
  - There is no explicit cleanup of generated files.
  - There are no file size, page count, or image dimension limits yet.

## Logic That Should Move to Backend Services

- `backend/app/services/pdf_service.py`
  - Validate uploaded PDF files.
  - Convert PDF bytes to page images.
  - Apply future limits for page count, file size, and conversion settings.
  - Hide Poppler-specific behavior behind a service interface.

- `backend/app/services/ocr_service.py`
  - Run Tesseract OCR against images.
  - Extract text from each PDF page image.
  - Extract text from uploaded image files.
  - Normalize and combine OCR text.
  - Centralize Tesseract configuration and error handling.

- `backend/app/services/llm_service.py`
  - Own Hugging Face client creation.
  - Read model name and API key from backend settings.
  - Build the document extraction prompt.
  - Call the model and return the raw model response.
  - Handle LLM errors, timeouts, token limits, and response validation.

- `backend/app/services/json_service.py`
  - Clean model output.
  - Remove markdown fences.
  - Isolate JSON candidates.
  - Parse JSON safely.
  - Validate the response shape before returning it to the API.

- Future API layer:
  - Add a FastAPI upload endpoint only after the service extraction is complete and tested.
  - Keep request/response schemas in `backend/app/schemas`.
  - Keep endpoint routing under `backend/app/api`.

## Logic That Should Stay in UI

For now, Streamlit should remain responsible for the interactive user experience:

- File uploader.
- Document/page preview.
- Editable OCR text review area.
- Analyze button.
- Progress spinners and user-facing errors.
- Display of extracted JSON.
- Download button for generated output.

During the transition, Streamlit can either continue calling local helper functions or later call the FastAPI backend, but the existing app should remain runnable until a dedicated migration task changes that behavior.

## Recommended Refactor Order

1. Extract PDF utilities into a backend service without changing Streamlit behavior.
2. Extract OCR utilities into a backend service and compare output against the current app.
3. Extract LLM cleanup and extraction logic into a backend service using environment-based settings.
4. Extract JSON generation and parsing helpers into a backend service.
5. Add a FastAPI upload endpoint that uses the extracted services.
6. Test the backend endpoint with PDF and image uploads.
7. Keep Streamlit working during the transition, then decide whether it should call the backend API or remain a temporary local UI.

## Risks / Things To Be Careful About

- Do not hardcode API keys, tokens, secrets, database passwords, or Azure credentials.
- Preserve existing app behavior while moving logic in small steps.
- Tesseract must be installed and available on the host system.
- Poppler must be installed and available for `pdf2image` PDF conversion.
- Large PDFs can create many images and consume significant memory.
- Add page count, file size, and timeout limits before exposing upload endpoints broadly.
- Handle OCR failures, PDF conversion failures, invalid image files, LLM API failures, and invalid JSON responses clearly.
- Clean up temporary files and generated PDFs after each request.
- Avoid shared fixed filenames when multiple users or requests are introduced.
- Keep the Streamlit app runnable until a dedicated task replaces or removes it.

## No Code Changes Yet

This task only creates a planning document. It does not refactor `paperSleuth.py`, create backend service files, add dependencies, add database code, add authentication, add Azure Blob Storage, add Docker, or add frontend code.
