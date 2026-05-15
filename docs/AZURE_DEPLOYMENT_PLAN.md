# PaperSleuth Azure Deployment Plan

## Target Architecture

PaperSleuth is moving from a local Streamlit OCR app toward a SaaS architecture with a FastAPI backend and React frontend.

Recommended Azure target:

- Backend API: Azure App Service or Azure Container Apps.
- Frontend: Azure Static Web Apps.
- Database: Azure Database for PostgreSQL.
- File storage: Azure Blob Storage for uploaded PDFs, extracted text, and JSON output.
- Secrets/configuration: Azure App Settings for basic configuration, with Azure Key Vault for production secrets when ready.

## Backend Runtime

The backend currently runs FastAPI with local SQLite and local filesystem storage. For Azure, the backend should run with:

- Python 3.11.
- Tesseract OCR installed.
- Poppler installed for PDF conversion.
- Environment variables configured in Azure App Settings or Container Apps secrets.

If using Azure Container Apps, the backend Dockerfile can carry OCR system dependencies. If using App Service without a custom container, confirm Tesseract and Poppler availability first.

## Frontend Runtime

The React/Vite frontend can be built and deployed to Azure Static Web Apps.

The frontend should set:

```text
VITE_API_BASE_URL=https://<backend-hostname>
```

## Database

SQLite is only for local development.

For Azure, use Azure Database for PostgreSQL and configure:

```text
DATABASE_URL=postgresql://<user>:<password>@<host>:5432/<database>
```

Do not commit production database credentials.

## Storage

Local storage currently writes files under `backend/storage/`.

Azure Blob Storage should later replace the local storage implementation behind `backend/app/services/storage_service.py`.

Future storage settings:

```text
STORAGE_PROVIDER=azure
AZURE_STORAGE_CONNECTION_STRING=<from Azure>
AZURE_STORAGE_CONTAINER_NAME=papersleuth-documents
```

Do not commit Azure connection strings.

## Required Environment Variables

```text
DATABASE_URL
SECRET_KEY
ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES
BACKEND_CORS_ORIGINS
STORAGE_PROVIDER
AZURE_STORAGE_CONNECTION_STRING
AZURE_STORAGE_CONTAINER_NAME
HF_API_KEY
HF_MODEL_NAME
LLM_API_KEY
```

`HF_API_KEY` is used by the current Hugging Face integration. `LLM_API_KEY` is reserved as a generic future name if the LLM provider changes.

## Migration Steps

1. Keep the local SQLite and filesystem workflow stable.
2. Add Alembic migrations before relying on PostgreSQL in shared environments.
3. Add an Azure Blob implementation behind the storage service helpers.
4. Deploy the backend with environment variables and OCR system dependencies.
5. Deploy the frontend with `VITE_API_BASE_URL` pointing to the backend.
6. Run an end-to-end test: signup, login, upload PDF, view document history, view extracted text, view JSON output.

## Security Notes

- Replace the default `SECRET_KEY` before any hosted deployment.
- Store secrets in Azure App Settings or Key Vault.
- Do not commit `.env`, database files, storage files, or Azure credentials.
- Add HTTPS-only production CORS origins before exposing the API publicly.
