import logging
import io
import os
import json
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from supabase import create_client
from PyPDF2 import PdfReader
from fastapi import UploadFile
from models import SearchRequest, SearchResult
from typing import Dict

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FAISS & Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL,device="cpu")
dimension = 384
index = faiss.IndexFlatL2(dimension)

# In-memory storage for metadata
pdf_texts: List[Dict[str, str]] = []


# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA256 hash of a file's content.
    Used for duplicate detection before uploading.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def get_pdf_signed_url(file_path: str) -> str:
    """
    Create a time-limited signed URL for a PDF stored in Supabase Storage.
    """
    try:
        response = supabase.storage.from_(BUCKET_NAME).create_signed_url(
            file_path, 3600
        )
        return response.get("signedURL") or response.get("signedUrl") or ""
    except Exception as e:
        logging.warning(f"Failed to create signed URL for {file_path}: {e}")
        return ""


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
    # Read file bytes
    file_bytes: bytes = file.file.read()
    file.file.seek(0)  # reset pointer

    # 1. Compute hash for duplicate detection
    file_hash: str = compute_file_hash(file_bytes)
    try:
        existing = supabase.table("documents").select("*").eq("file_hash", file_hash).execute()
        if existing.data:
            return {"message": f"{file.filename} already uploaded."}
    except Exception as e:
        logging.error(f"Duplicate check failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check for duplicates. Please try again.")

    # 2. Extract text from PDF
    text: str = extract_text_from_pdf(file)
    if not text.strip():
        return {"message": "No extractable text found in PDF."}

    # 3. Generate embeddings
    try:
        embedding: np.ndarray = model.encode([text])
    except Exception as e:
        logging.error(f"Embedding generation failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document. Please try again.")

    file_path: str = f"{file.filename}"

    # 4. Upload to Supabase Storage (first external modification)
    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            file=file_bytes, path=file_path, file_options={"upsert": "false"}
        )
    except Exception as e:
        logging.error(f"Storage upload failed for {file.filename}: {e}")
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
            supabase.storage.from_(BUCKET_NAME).remove([file_path])
            rollback_success = True
        except Exception as rollback_error:
            # Rollback itself failed - critical error
            logging.critical(
                f"CRITICAL: Upload rollback failed for {file.filename}. "
                f"DB error: {db_error}. Rollback error: {rollback_error}. "
                f"Orphaned file may exist in storage."
            )
        
        # Raise appropriate error based on rollback outcome
        if rollback_success:
            logging.error(f"DB insert failed for {file.filename}, rolled back storage upload: {db_error}")
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
        logging.error(
            f"FAISS indexing failed for {file.filename}: {e}. "
            f"File is stored in database and storage. Index will be rebuilt on restart."
        )
        raise HTTPException(
            status_code=500,
            detail="Upload complete but indexing failed. File will be indexed on next restart."
        )

    logging.info(f"Uploaded and indexed PDF: {file.filename}")
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
            logging.info("No documents found in Supabase DB. Starting with empty index.")
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

        logging.info(f"Loaded {len(docs)} documents into FAISS.")

    except Exception as e:
        logging.error(
            f"Failed to load index from Supabase during startup: {type(e).__name__}: {e}. "
            f"Starting with empty FAISS index."
        )
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
        return []

    # Encode query and search in FAISS
    query_embedding: np.ndarray = model.encode([request.query])
    D, I = index.search(query_embedding, request.top_k)

    results: List[SearchResult] = []
    for rank, idx in enumerate(I[0]):
        if idx == -1 or idx >= len(pdf_texts):
            continue
        file_info = pdf_texts[idx]
        snippet: str = file_info["content"][:200] + "..."
        file_path: str = file_info.get("file_path") or file_info["filename"]
        results.append(
            SearchResult(
                filename=file_info["filename"],
                snippet=snippet,
                score=float(D[0][rank]),
                url=get_pdf_signed_url(file_path),
            )
        )

    return results