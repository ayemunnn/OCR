# 🧠 PaperSleuth

PaperSleuth is a Streamlit app that extracts structured data from scanned documents using OCR (Tesseract) and the Mistral-Small-3.1-24B-Instruct-2503 LLM via Hugging Face.

## 📦 Features
- Upload PDFs or images
- Extract text using Tesseract
- Call Mistral OCR 2503 for structured JSON extraction
- Download output as a PDF

## 🚀 Setup

### Clone and install dependencies

```bash
git clone https://github.com/your-username/paperSleuth.git
cd paperSleuth
pip install -r requirements.txt
```

## FastAPI Backend

Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Run the backend API:

```bash
uvicorn app.main:app --reload
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

To test PDF processing, open Swagger UI, expand `POST /documents/process`, upload a PDF file, and execute the request.

If `HF_API_KEY` is not configured, the endpoint still runs OCR and returns:

```text
LLM processing skipped because API key is not configured.
```

### Local SQLite schema changes

This project does not use Alembic yet. During local development, if a model change adds columns or tables and the backend fails because the existing SQLite schema is stale, stop the backend and delete `backend/papersleuth.db`. The app will recreate the local database on startup.
