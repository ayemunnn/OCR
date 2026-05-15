# AGENTS.md

## Project Overview

PaperSleuth is being upgraded from a Streamlit OCR app into an Azure-hosted SaaS product.

The current app processes scanned PDFs using OCR and LLM-based cleanup/extraction. The SaaS version will eventually include authentication, user-specific document history, uploaded PDF storage, JSON output storage, and Azure deployment.

## Current Development Approach

Work in small, safe, reviewable steps.

The existing Streamlit app is currently working and must not be broken while the SaaS architecture is added gradually.

## Important Rules

- Do not delete the existing Streamlit app.
- Do not rename the existing main Streamlit file unless explicitly asked.
- Do not move OCR logic until a dedicated refactor task is given.
- Do not add authentication until a dedicated auth task is given.
- Do not add database models until a dedicated database task is given.
- Do not add Azure Blob Storage until a dedicated storage task is given.
- Do not add Docker until a dedicated Docker task is given.
- Do not hardcode API keys, tokens, secrets, database passwords, or Azure credentials.
- Use environment variables for secrets.
- Keep changes small and easy to review.
- Prefer clear folder structure over clever abstractions.
- Use simple, beginner-friendly code where possible.

## Target Architecture

Long-term target structure:

backend/
  app/
    main.py
    api/
    core/
    services/
    models/
    schemas/
    db/
  tests/

frontend/

infra/

docs/

The backend should use FastAPI.

The frontend may later use React or Next.js.

The deployment target is Azure.

## Local Development Expectations

The user is developing locally on Windows using Miniconda.

Use Python 3.11 unless there is a strong reason not to.

Expected local environment:

```bash
conda create -n papersleuth python=3.11 -y
conda activate papersleuth
```
