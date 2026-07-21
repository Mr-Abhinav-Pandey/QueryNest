import logging
import io
import json
import hashlib
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from supabase import create_client
from PyPDF2 import PdfReader
from fastapi import UploadFile, HTTPException
from models import SearchRequest, SearchResult
from typing import Dict
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_BUCKET,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    SIGNED_URL_TTL_SECONDS,
)


logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase client initialized")

# FAISS & Embedding Model
model = SentenceTransformer(EMBEDDING_MODEL,device="cpu")
dimension = EMBEDDING_DIMENSION
index = faiss.IndexFlatL2(dimension)
logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
logger.info("FAISS index initialized: dimension=%s", dimension)

# In-memory storage for metadata
pdf_texts: List[Dict[str, str]] = []

from datetime import datetime, timedelta
signed_url_cache: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA256 hash of a file's content.
    Used for duplicate detection before uploading.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def get_pdf_signed_url(file_path: str) -> Optional[str]:
    """
    Return a signed URL for a PDF.
    Uses an in-memory cache until the URL expires.
    """

    now = datetime.utcnow()

    cached = signed_url_cache.get(file_path)

    if cached and cached["expires"] > now:
        logger.debug("Signed URL cache hit: %s", file_path)
        return cached["url"]

    try:
        response = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(
            file_path,
            SIGNED_URL_TTL_SECONDS,
        )

        url = response.get("signedURL") or response.get("signedUrl")

        if url:
            signed_url_cache[file_path] = {
                "url": url,
                "expires": now + timedelta(seconds=SIGNED_URL_TTL_SECONDS - 30),
            }

        return url

    except Exception as e:
        logger.warning(
            "Failed to create signed URL for %s: %s",
            file_path,
            e,
        )
        return None


def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract all text from a given PDF file.
    Returns concatenated text of all pages.
    """
    reader = PdfReader(file.file)
    text: str = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# --------------------------------------------------------
# Uploading & Indexing Functions
# --------------------------------------------------------

def upload_and_index(file: UploadFile) -> Dict[str, str]:
    """
    Upload a PDF file to Supabase storage, extract text, generate embeddings,
    store metadata in database, and update FAISS index.
    
    Order: Storage → Database → FAISS (to ensure consistency).
    If DB insert fails, rolls back storage upload.
    If FAISS insertion fails, preserves storage and DB data.
    """
    logger.info("Upload started: filename=%s", file.filename)

    # Read file bytes
    file_bytes: bytes = file.file.read()
    file.file.seek(0)  # reset pointer

    # 1. Compute hash for duplicate detection
    file_hash: str = compute_file_hash(file_bytes)
    try:
        existing = supabase.table("documents").select("*").eq("file_hash", file_hash).execute()
        if existing.data:
            logger.info("Duplicate document detected: filename=%s", file.filename)
            return {"message": f"{file.filename} already uploaded."}
    except Exception as e:
        logger.exception("Duplicate check failed: filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Failed to check for duplicates. Please try again.")

    # 2. Extract text from PDF
    text: str = extract_text_from_pdf(file)
    if not text.strip():
        logger.warning("No extractable text found: filename=%s", file.filename)
        return {"message": "No extractable text found in PDF."}

    # 3. Generate embeddings
    try:
        embedding: np.ndarray = model.encode([text])
    except Exception as e:
        logger.exception("Embedding generation failed: filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Failed to process document. Please try again.")

    file_path: str = f"{file.filename}"

    # 4. Upload to Supabase Storage (first external modification)
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            file=file_bytes, path=file_path, file_options={"upsert": "false"}
        )
    except Exception as e:
        logger.exception("Storage upload failed: filename=%s", file.filename)
        raise HTTPException(status_code=500, detail="Failed to store document. Please try again.")

    # 5. Insert metadata into Supabase DB (second external modification, can be rolled back)
    try:
        supabase.table("documents").insert({
            "filename": file.filename,
            "file_path": file_path,
            "content": text,
            "embedding": embedding[0].tolist(),
            "file_hash": file_hash
        }).execute()
    except Exception as db_error:
        # Rollback: delete file from storage
        rollback_success = False
        try:
            supabase.storage.from_(SUPABASE_BUCKET).remove([file_path])
            rollback_success = True
        except Exception as rollback_error:
            # Rollback itself failed - critical error
            logger.exception(
                "Upload rollback failed: filename=%s", file.filename
            )
        
        # Raise appropriate error based on rollback outcome
        if rollback_success:
            logger.exception("Database insert failed after storage upload: filename=%s", file.filename)
            raise HTTPException(status_code=500, detail="Upload failed. Please try again.")
        else:
            raise HTTPException(
                status_code=500,
                detail="Upload failed during cleanup. Please retry."
            )

    # 6. Add to FAISS index (third modification, not rolled back on failure)
    try:
        index.add(embedding)
        pdf_texts.append({
            "filename": file.filename,
            "content": text,
            "file_path": file_path,
        })
    except Exception as e:
        logger.exception("FAISS indexing failed: filename=%s", file.filename)
        raise HTTPException(
            status_code=500,
            detail="Upload complete but indexing failed. File will be indexed on next restart."
        )

    logger.info("Upload completed successfully: filename=%s", file.filename)
    return {"message": f"File {file.filename} uploaded and indexed successfully."}


# --------------------------------------------------------
# Loading Data into FAISS
# --------------------------------------------------------

def load_index_from_db() -> None:
    """
    Load stored embeddings and document metadata
    from Supabase database into FAISS and memory.
    Gracefully handles Supabase connection failures.
    """
    global index, pdf_texts

    logger.info("Loading indexed documents from Supabase")

    # Reset FAISS and local memory
    pdf_texts.clear()
    index.reset()

    try:
        # Fetch all stored documents
        response = supabase.table("documents").select(
            "filename, content, embedding, file_path"
        ).execute()
        docs: List[Dict[str, Any]] = response.data or []

        if not docs:
            logger.info("No documents found in Supabase DB")
            return

        embeddings: List[List[float]] = []
        for doc in docs:
            pdf_texts.append({
                "filename": doc["filename"],
                "content": doc["content"],
                "file_path": doc.get("file_path") or doc["filename"],
            })
            embedding = doc["embedding"]
            # Ensure embedding is a list of floats
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            embeddings.append(embedding)

        # Add embeddings into FAISS
        if embeddings:
            embeddings_np: np.ndarray = np.array(embeddings).astype("float32")
            index.add(embeddings_np)

        logger.info("Loaded indexed documents: count=%s", len(docs))

    except Exception as e:
        logger.exception("Failed to load index from Supabase during startup")
        # App continues with empty index


# --------------------------------------------------------
# Semantic Search
# --------------------------------------------------------

def semantic_search(request: SearchRequest) -> List[SearchResult]:
    """
    Perform semantic search on indexed documents.
    Returns a list of SearchResult with filename, snippet, score, and signed PDF URL.
    """
    if len(pdf_texts) == 0:
        logger.info("Search request received with empty index: top_k=%s", request.top_k)
        return []

    # Encode query and search in FAISS
    logger.info("Search request received: top_k=%s", request.top_k)
    query_embedding: np.ndarray = model.encode([request.query])
    D, I = index.search(query_embedding, request.top_k)
    MAX_DISTANCE = 1.20
    results: List[SearchResult] = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue

        distance = float(D[0][rank])

        if distance > MAX_DISTANCE:
            logger.debug(
                "Skipping low-confidence result: %.4f",
                distance,
            )
            continue
        file_info = pdf_texts[idx]
        snippet: str = file_info["content"][:200] + "..."
        file_path: str = file_info.get("file_path") or file_info["filename"]
        results.append(
            SearchResult(
                filename=file_info["filename"],
                snippet=snippet,
                score=round(distance, 4),
                url=get_pdf_signed_url(file_path),
            )
        )

    return results


def semantic_search_instrumented(request: SearchRequest) -> (List[SearchResult], Dict[str, float]):
    """
    Instrumented version of semantic_search that returns timing breakdowns:
    - embedding_time: time to encode the query
    - faiss_time: time spent in FAISS search
    - assembly_time: time to assemble results (including signed URL generation)
    - total_time: end-to-end time

    Returns (results, timings)
    """
    timings: Dict[str, float] = {
        "embedding_time": 0.0,
        "faiss_time": 0.0,
        "assembly_time": 0.0,
        "total_time": 0.0,
    }

    import time

    start_total = time.perf_counter()

    if len(pdf_texts) == 0:
        timings["total_time"] = time.perf_counter() - start_total
        return [], timings

    # 1) Encode query
    t0 = time.perf_counter()
    query_embedding: np.ndarray = model.encode([request.query])
    t1 = time.perf_counter()
    timings["embedding_time"] = t1 - t0

    # 2) FAISS search
    t0 = time.perf_counter()
    D, I = index.search(query_embedding, request.top_k)
    t1 = time.perf_counter()
    timings["faiss_time"] = t1 - t0

    # 3) Assemble results (includes signed URL generation)
    t0 = time.perf_counter()
    results: List[SearchResult] = []
    for rank, idx in enumerate(I[0]):
        if idx == -1 or idx >= len(pdf_texts):
            continue
        file_info = pdf_texts[idx]
        snippet: str = file_info["content"][:200] + "..."
        file_path: str = file_info.get("file_path") or file_info["filename"]
        # Call get_pdf_signed_url (may return None if fails)
        signed = None
        try:
            signed = get_pdf_signed_url(file_path)
        except Exception:
            signed = None

        results.append(
            SearchResult(
                filename=file_info["filename"],
                snippet=snippet,
                score=float(D[0][rank]),
                url=signed,
            )
        )
    t1 = time.perf_counter()
    timings["assembly_time"] = t1 - t0

    timings["total_time"] = time.perf_counter() - start_total
    return results, timings