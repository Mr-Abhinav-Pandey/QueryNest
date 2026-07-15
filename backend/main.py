import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from logging_config import configure_logging


configure_logging()

from config import (
    APP_ENVIRONMENT,
    APPLICATION_NAME,
    APPLICATION_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    MAX_UPLOAD_SIZE_MB,
    MAX_UPLOAD_SIZE_BYTES,
)
from functions import upload_and_index, semantic_search, load_index_from_db
import functions as backend_functions
from models import (
    HealthResponse,
    ReadinessFailureResponse,
    ReadinessResponse,
    SearchRequest,
    SearchResult,
    VersionResponse,
)


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
    app.state.startup_time = datetime.now(timezone.utc).isoformat()
    load_index_from_db()
    app.state.runtime_ready = True
    logger.info("Application startup complete")
    yield
    # --- Runs on shutdown ---
    logger.info("Application shutdown")


app = FastAPI(
    title="QueryNest Backend",
    description="Semantic PDF search using FastAPI, FAISS, and Supabase.",
    version=APPLICATION_VERSION,
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
    if (
        not file.filename.lower().endswith(".pdf")
        or file.content_type != "application/pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Validate upload size
    contents = await file.read()

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds maximum upload size of {MAX_UPLOAD_SIZE_MB} MB."
        )

    # Reset stream so upload_and_index() can read the file again
    await file.seek(0)

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


def _get_readiness_checks(app: FastAPI) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "supabase_client_initialized": backend_functions.supabase is not None,
        "sentence_transformer_loaded": backend_functions.model is not None,
        "faiss_index_initialized": backend_functions.index is not None,
        "runtime_state_available": bool(getattr(app.state, "runtime_ready", False)),
        "startup_time_available": bool(getattr(app.state, "startup_time", None)),
    }
    issues = [name for name, passed in checks.items() if not passed]
    return checks, issues


@app.get("/health", tags=["Health Check"], response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify API is running.

    Returns:
        HealthResponse: Liveness status.
    """
    return HealthResponse(status="healthy")


@app.get("/ready", tags=["Health Check"], response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse | JSONResponse:
    """
    Readiness probe for production traffic routing.
    """
    checks, issues = _get_readiness_checks(app)
    if issues:
        reason = "Readiness check failed: " + ", ".join(issues)
        logger.warning(reason)
        return JSONResponse(
            status_code=503,
            content=ReadinessFailureResponse(
                status="not_ready",
                reason=reason,
                checks=checks,
            ).model_dump(),
        )

    logger.info("Readiness check passed")
    return ReadinessResponse(
        status="ready",
        documents_indexed=len(backend_functions.pdf_texts),
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        faiss_vectors=int(backend_functions.index.ntotal),
    )


@app.get("/version", tags=["Operational"], response_model=VersionResponse)
async def version_info() -> VersionResponse:
    """
    Operational metadata endpoint for deployment diagnostics.
    """
    startup_time = getattr(app.state, "startup_time", None)
    if not startup_time:
        startup_time = datetime.now(timezone.utc).isoformat()

    from fastapi import __version__ as fastapi_version
    import platform

    return VersionResponse(
        application=APPLICATION_NAME,
        version=APPLICATION_VERSION,
        python_version=platform.python_version(),
        fastapi_version=fastapi_version,
        startup_time=startup_time,
        environment=APP_ENVIRONMENT,
    )


@app.get("/", tags=["Health Check"])
async def root_health_check() -> Dict[str, str]:
    """Backward-compatible root health response."""
    return {"status": "ok"}