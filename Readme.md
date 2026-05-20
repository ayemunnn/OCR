# PaperSleuth

PaperSleuth is being upgraded from a working Streamlit OCR app into a local SaaS MVP with a FastAPI backend and React frontend.

The original Streamlit app, `paperSleuth.py`, is preserved during the migration and should not be deleted or broken.

## Current MVP

- FastAPI backend with health, auth, document processing, and document history endpoints.
- JWT authentication with signup, login, and `/auth/me`.
- User-owned document records stored in SQLite for direct local runs or PostgreSQL through Docker Compose.
- Local file storage under `backend/storage/` for uploaded PDFs, OCR text, and JSON output.
- React/Vite frontend for signup, login, PDF upload, history, extracted text, and JSON viewing.
- CORS configured for local frontend development.

## Local Backend Setup With Conda

```bash
conda create -n papersleuth python=3.11 -y
conda activate papersleuth
cd backend
pip install -r requirements.txt
alembic -c alembic.ini upgrade head
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

## Optional LLM Setup

PaperSleuth can process PDFs in OCR-only mode. To enable LLM cleanup and JSON extraction, provide a Hugging Face token through environment variables.

For direct Miniconda/local backend runs, copy `backend/.env.example` to `backend/.env` and set:

```env
HF_API_KEY=your_hugging_face_token_here
HF_MODEL_NAME=google/gemma-3-27b-it
```

For Docker Compose runs, set the variable in your shell before starting the stack:

```powershell
$env:HF_API_KEY="your_hugging_face_token_here"
docker compose up --build
```

Do not commit real API keys. If `HF_API_KEY` is blank, uploads still complete with OCR text and the response will say that LLM processing was skipped.

## Docker Workflow

Docker is optional. The normal Miniconda workflow remains supported.

Build and run the backend, frontend, and PostgreSQL together:

```bash
docker compose up --build
```

Open the frontend at `http://127.0.0.1:5173` and the backend API docs at `http://127.0.0.1:8000/docs`.

The Docker backend uses PostgreSQL with local development credentials from `docker-compose.yml`. These credentials are not production secrets.

Apply database migrations to the Docker PostgreSQL database:

```bash
docker compose exec backend alembic -c alembic.ini upgrade head
```

Stop the stack:

```bash
docker compose down
```

Reset the local Docker PostgreSQL database volume:

```bash
docker compose down -v
```

Direct Miniconda/local backend runs still use SQLite unless `DATABASE_URL` is set.

## Database Migrations

SQLite is for local development only. Production or Azure deployments should use PostgreSQL, ideally Azure Database for PostgreSQL.

PaperSleuth uses Alembic for database migrations. Run migrations from the `backend/` folder:

```bash
alembic -c alembic.ini upgrade head
```

Create a future migration after model changes with:

```bash
alembic -c alembic.ini revision --autogenerate -m "describe change"
```

Then review the generated migration before applying it.

Alembic is preferred for schema changes going forward. The old `init_db()` startup hook is kept as a no-op compatibility shim and does not create tables.

If your existing local SQLite database was created before Alembic was added, stop the backend, delete:

```text
backend/papersleuth.db
```

Then run:

```bash
cd backend
alembic -c alembic.ini upgrade head
```

Docker does not run migrations automatically. After starting the Docker stack, run:

```bash
docker compose exec backend alembic -c alembic.ini upgrade head
```

## PostgreSQL Readiness

SQLite remains the default for direct Miniconda/local development. Docker Compose uses PostgreSQL to match production-style database usage. You can also run the backend directly against PostgreSQL by setting `DATABASE_URL`, for example:

```text
DATABASE_URL=postgresql+psycopg2://<username>:<password>@localhost:5432/<database_name>
```

Then run Alembic against that `DATABASE_URL` before starting the backend.

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

## Azure Deployment

Azure deployment is not automated yet. Before deploying, review:

```text
docs/AZURE_DEPLOYMENT_PLAN.md
```

The plan covers the recommended Azure architecture, production environment variables, deployment checklist, migration guidance, and security notes.
