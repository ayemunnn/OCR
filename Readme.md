# PaperSleuth

PaperSleuth is being upgraded from a working Streamlit OCR app into a local SaaS MVP with a FastAPI backend and React frontend.

The original Streamlit app, `paperSleuth.py`, is preserved during the migration and should not be deleted or broken.

## Current MVP

- FastAPI backend with health, auth, document processing, and document history endpoints.
- JWT authentication with signup, login, and `/auth/me`.
- User-owned document records stored in SQLite.
- Local file storage under `backend/storage/` for uploaded PDFs, OCR text, and JSON output.
- React/Vite frontend for signup, login, PDF upload, history, extracted text, and JSON viewing.
- CORS configured for local frontend development.

## Local Backend Setup With Conda

```bash
conda create -n papersleuth python=3.11 -y
conda activate papersleuth
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

The backend requires OCR system tools for full PDF processing:

- Tesseract OCR
- Poppler

## Frontend Development

The React frontend expects the FastAPI backend to be running at `http://127.0.0.1:8000`.

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

To point the frontend at a different backend, create a frontend environment file with:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## End-to-End Test Flow

1. Start the backend.
2. Start the frontend.
3. Open `http://127.0.0.1:5173`.
4. Sign up for a local user account.
5. Log in.
6. Upload a PDF.
7. View document history.
8. Select a document.
9. View extracted text.
10. View JSON output if LLM output is available.

If `HF_API_KEY` is not configured, the backend still runs OCR and reports that LLM processing was skipped.

## Docker Workflow

Docker is optional. The normal Miniconda workflow remains supported.

Build and run the backend and frontend together with SQLite:

```bash
docker compose up --build
```

Open the frontend at `http://127.0.0.1:5173` and the backend API docs at `http://127.0.0.1:8000/docs`.

Run the optional PostgreSQL service for local experimentation:

```bash
docker compose --profile postgres up postgres
```

To use PostgreSQL with the backend, set `DATABASE_URL` to a PostgreSQL URL before starting the backend. SQLite remains the default for local development.

## Database Notes

SQLite is for local development only. Production or Azure deployments should use PostgreSQL, ideally Azure Database for PostgreSQL.

This project does not use Alembic yet. During local development, if a model change adds columns or tables and the backend fails because the existing SQLite schema is stale, stop the backend and delete:

```text
backend/papersleuth.db
```

The app will recreate the local database on startup.

## Storage Providers

Local filesystem storage is the default and writes processed files under `backend/storage/`.

```text
STORAGE_PROVIDER=local
```

Azure Blob Storage can be enabled later without changing the document API:

```text
STORAGE_PROVIDER=azure
AZURE_STORAGE_CONNECTION_STRING=<your Azure connection string>
AZURE_STORAGE_CONTAINER_NAME=papersleuth-documents
```

Do not commit real Azure credentials. If `STORAGE_PROVIDER=azure` is set without a connection string, the backend returns a clear configuration error.

## Ignored Local Files

These should not be committed:

- `.env`
- `backend/storage/`
- `backend/papersleuth.db`
- generated PDF/log/cache files

Use `.env.example` files only for documenting configuration.

## Azure Readiness

See:

```text
docs/AZURE_DEPLOYMENT_PLAN.md
```

Planned Azure services:

- Azure App Service or Azure Container Apps for the backend.
- Azure Static Web Apps for the frontend.
- Azure Database for PostgreSQL.
- Azure Blob Storage.
- Azure App Settings or Key Vault for secrets.
