# PaperSleuth Azure Deployment Plan

This document prepares PaperSleuth for Azure deployment. It is planning and configuration guidance only; it does not deploy anything.

## Recommended Azure Architecture

PaperSleuth should deploy as separate frontend, backend, database, and storage services:

- Frontend: Azure Static Web Apps for the React/Vite app.
- Backend: Azure Container Apps or Azure App Service for the FastAPI API.
- Database: Azure Database for PostgreSQL.
- File storage: Azure Blob Storage for uploaded PDFs, extracted OCR text, and JSON outputs.
- Secrets/configuration: Azure App Settings for environment variables, with Azure Key Vault for production secrets when ready.
- Container image: Azure Container Registry if using Azure Container Apps or custom-container App Service.

Azure Container Apps is a strong backend target because the backend needs OCR system dependencies such as Tesseract and Poppler. Azure App Service is also possible, but verify OCR dependency support before choosing a non-container deployment.

## Production Environment Variables

### Backend

Configure these values in Azure App Settings, Container Apps secrets, or Key Vault references. Do not commit production values to the repository.

```text
APP_NAME=PaperSleuth
APP_ENV=production
APP_DEBUG=false
DATABASE_URL=postgresql+psycopg2://<username>:<password>@<host>:5432/<database_name>
SECRET_KEY=<strong-production-secret>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
BACKEND_CORS_ORIGINS=["https://<frontend-domain>"]
STORAGE_PROVIDER=azure
AZURE_STORAGE_CONNECTION_STRING=<azure-storage-connection-string>
AZURE_STORAGE_CONTAINER_NAME=papersleuth-documents
HF_API_KEY=<hugging-face-token-if-used>
LLM_API_KEY=<generic-llm-token-if-provider-changes>
HF_MODEL_NAME=google/gemma-3-27b-it
```

Notes:

- `DATABASE_URL` should point to Azure Database for PostgreSQL in production.
- `SECRET_KEY` must be changed from the local default before deployment.
- `BACKEND_CORS_ORIGINS` should include only the deployed frontend URL.
- Use either `HF_API_KEY` for the current Hugging Face integration or a future `LLM_API_KEY` if the provider changes.

### Frontend

Configure this during the Azure Static Web Apps build/deploy process:

```text
VITE_API_BASE_URL=https://<backend-api-domain>
```

## Deployment Checklist

1. Create an Azure Resource Group.
2. Create Azure Database for PostgreSQL.
3. Create an Azure Storage Account.
4. Create a Blob container for PaperSleuth documents.
5. Create Azure Container Registry if deploying the backend as a container.
6. Build and publish the backend container image if using Container Apps.
7. Create the backend hosting service with Azure Container Apps or Azure App Service.
8. Set backend environment variables in App Settings, Container Apps secrets, or Key Vault references.
9. Run Alembic migrations against the production PostgreSQL database.
10. Deploy the backend.
11. Deploy the frontend to Azure Static Web Apps.
12. Set `VITE_API_BASE_URL` to the backend API URL.
13. Restrict backend CORS to the production frontend URL.
14. Test signup, login, PDF upload, document history, extracted text, and JSON output.

## Database Migrations

PaperSleuth uses Alembic for schema migrations.

Before deploying the backend against a new Azure PostgreSQL database, run:

```bash
cd backend
alembic -c alembic.ini upgrade head
```

For hosted environments, run migrations from a trusted deployment shell, release job, or temporary admin environment that has the production `DATABASE_URL` configured. Do not rely on deleting databases or resetting schemas in production.

## Local vs Production

- SQLite is a local-only fallback for direct Miniconda development.
- PostgreSQL should be used in production.
- Local filesystem storage under `backend/storage/` is for development only.
- Azure Blob Storage should be used in production.
- `.env` files must not be committed.
- `backend/.env.example` should contain placeholders only, never real credentials.
- Docker Compose values are development-only and should not be reused as production secrets.

## Security Notes

- Never use the default `SECRET_KEY` in production.
- Do not commit `.env` files, database files, uploaded PDFs, generated outputs, API keys, or Azure credentials.
- Use a strong database password and rotate it if it is ever exposed.
- Restrict `BACKEND_CORS_ORIGINS` to the production frontend URL.
- Use HTTPS only in production.
- Store secrets in Azure App Settings or Azure Key Vault.
- Prefer Key Vault references for database passwords, storage connection strings, JWT secrets, and LLM API keys.
- Confirm Azure Blob containers are not public unless there is a deliberate product requirement.
- Review logs to ensure OCR text, document contents, and secrets are not accidentally logged.
