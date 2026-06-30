import logging
from contextlib import asynccontextmanager
from typing import Dict,List
from fastapi import FastAPI, HTTPException, UploadFile, File
from logging_config import configure_logging


configure_logging()

from functions import upload_and_index, semantic_search, load_index_from_db
from models import SearchRequest, SearchResult


logger = logging.getLogger(__name__)


# --------------------------------------------------------
# FastAPI Application with Lifespan
# --------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Runs on startup and shutdown.
    """
    # --- Runs on startup ---
    logger.info("Application startup: loading indexed documents")
    load_index_from_db()
    logger.info("Application startup complete")
    yield
    # --- Runs on shutdown ---
    logger.info("Application shutdown")


app = FastAPI(
    title="QueryNest Backend",
    description="Semantic PDF search using FastAPI, FAISS, and Supabase.",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------------
# API Routes
# --------------------------------------------------------

@app.post("/upload_pdf", tags=["PDF Management"])
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload a PDF file, extract text, create embeddings,
    store in Supabase, and index in FAISS.

    Returns:
        dict: Message about upload status.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported."
        )
    return upload_and_index(file)


@app.post("/search", tags=["Semantic Search"], response_model=List[SearchResult])
async def search_files(request: SearchRequest) -> List[SearchResult]:
    """
    Perform semantic search on uploaded PDFs.

    Args:
        request (SearchRequest): Search query and number of results.

    Returns:
        List[SearchResult]: List of matched PDFs with snippets.
    """
    results = semantic_search(request)
    if not results:
        raise HTTPException(status_code=404, detail="No PDFs uploaded yet.")
    return results


@app.get("/", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API is running.

    Returns:
        dict: {"status": "ok"}
    """
    return {"status": "ok"}