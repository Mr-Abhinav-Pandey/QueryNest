# QueryNest

QueryNest is a semantic document retrieval platform for PDF files. It lets users upload documents, extracts text, generates Sentence Transformers embeddings, and searches by meaning using FAISS vector search with Supabase-backed storage and metadata. The result is an information retrieval workflow focused on semantic search rather than filename matching.

---

## Key Concepts

- Semantic Search
- Information Retrieval
- Embeddings
- Vector Search
- Approximate Nearest Neighbor Retrieval
- REST APIs
- Document Processing

---

## Features

- PDF upload from a Flutter client
- Text extraction from PDFs using `PyPDF2`
- Sentence Transformer embeddings with `all-MiniLM-L6-v2`
- FAISS indexing and top-k retrieval
- Duplicate detection using SHA-256 file hashes
- Bulk document indexing from Supabase Storage
- Search benchmarking and profiling utilities
- Signed URLs for document access
- Supabase Storage and metadata persistence
- Startup corpus reconstruction from Supabase on backend launch
- Reliability improvements that prevent orphaned uploads on database failure
- Explicit error handling around Supabase operations and signed URL generation
- Cross-platform Flutter frontend for Android, iOS, Web, Linux, macOS, and Windows

---

## Architecture

```text
PDF Upload
    ↓
Text Extraction (PyPDF2)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
SHA-256 Duplicate Detection
    ↓
FAISS Vector Index
    ↓
Supabase Storage + Metadata
    ↓
Semantic Search
    ↓
Top-k Retrieval
    ↓
Signed URL Generation
    ↓
Client Response
```

---

## Tech Stack

**Frontend**
- Flutter
- Dart
- `file_picker`
- `url_launcher`

**Backend**
- FastAPI
- Python
- Pydantic
- Supabase Python client

**Information Retrieval**
- Sentence Transformers
- Embeddings
- Semantic search
- Query validation

**Vector Search**
- FAISS
- In-memory vector index with startup reload from Supabase

**Storage**
- Supabase Storage
- Supabase table for document text, embeddings, and file hashes

**Document Processing**
- `PyPDF2`
- SHA-256 file hashing

---

## Repository Structure

```text
QueryNest/
├── backend/
│   ├── main.py
│   ├── functions.py
│   ├── models.py
│   ├── bench_search.py
│   ├── bulk_index_from_storage.py
│   └── requirements.txt
├── frontend/
│   ├── lib/
│   │   ├── main.dart
│   │   ├── api_service.dart
│   │   └── utils.dart
│   ├── android/
│   ├── ios/
│   ├── linux/
│   ├── macos/
│   ├── web/
│   └── windows/
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check endpoint |
| `POST` | `/upload_pdf` | Upload a PDF, extract text, generate embeddings, store metadata in Supabase, and index the document in FAISS |
| `POST` | `/search` | Run semantic search and return top-k matching documents with snippets and signed URLs |

---

## Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
flutter pub get
flutter run -d chrome
```

### Environment Variables

Copy `.env.example` to `backend/.env` and set:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET`

The Flutter client points to `http://localhost:8000` in `frontend/lib/api_service.dart`.

---
## Performance and Reliability Improvements

- Duplicate prevention through file hashes before upload
- Upload consistency with rollback if Supabase metadata insertion fails
- Startup restoration that reloads embeddings and metadata into the FAISS index
- Explicit error handling for Supabase storage, database, and signed URL failures
- Benchmarking utilities that measure embedding, FAISS, result assembly, and total search time

---

## Future Improvements

- Persistent FAISS indices for faster startup and reduced cold-start latency
- Relevance scores and ranking confidence
- Metadata-based filtering and pagination
- Support for additional document formats
- Automated tests and CI pipelines
- Additional document formats

---

## License

MIT License. See [LICENSE](LICENSE) for details.
