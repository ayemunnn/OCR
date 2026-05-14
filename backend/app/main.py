from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.documents import router as documents_router
from .db.init_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="PaperSleuth API",
    description="API backend for PaperSleuth OCR and document extraction SaaS.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)


@app.get("/")
def root():
    return {"message": "PaperSleuth API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
