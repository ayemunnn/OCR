from fastapi import FastAPI

from .api.documents import router as documents_router

app = FastAPI(
    title="PaperSleuth API",
    description="API backend for PaperSleuth OCR and document extraction SaaS.",
    version="0.1.0",
)

app.include_router(documents_router)


@app.get("/")
def root():
    return {"message": "PaperSleuth API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
