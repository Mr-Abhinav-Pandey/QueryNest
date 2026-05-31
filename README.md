# QueryNest

QueryNest is a semantic PDF search application. Instead of searching files by name, you ask natural-language questions and retrieve the most relevant documents based on meaning.

## Features

- Upload and manage PDF files from the client
- Semantic search powered by embeddings and vector similarity
- Natural-language queries to retrieve matching documents
- Cross-platform Flutter client (Android, iOS, Web, Desktop)

## Architecture

```
Flutter Client  →  FastAPI Backend  →  FAISS (in-memory index)
                              ↓
                         Supabase (storage + metadata)
```

| Layer    | Technology |
|----------|------------|
| Frontend | Flutter, Dart |
| Backend  | FastAPI, Python |
| Search   | Sentence Transformers (`all-MiniLM-L6-v2`), FAISS |
| Storage  | Supabase (PDF bucket + `documents` table) |

## Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) (>= 3.x)
- Python 3.10+ with `venv` support
- A [Supabase](https://supabase.com/) project with:
  - A storage bucket for PDFs
  - A `documents` table for file metadata and embeddings

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/QueryNest.git
cd QueryNest
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the environment template from the repository root and fill in your Supabase values:

```bash
# From the repository root
cp .env.example backend/.env   # macOS / Linux
copy ..\.env.example .env      # Windows (run from backend/)
```

Required variables:

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (keep secret) |
| `SUPABASE_BUCKET` | Storage bucket name for PDF uploads |

Start the API server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Health check: `GET /`.

### 3. Frontend setup

From the repository root:

```bash
cd frontend
flutter pub get
flutter run -d chrome
```

The Flutter client expects the backend at `http://localhost:8000` by default (see `frontend/lib/api_service.dart`).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload_pdf` | Upload a PDF for indexing |
| `POST` | `/search` | Semantic search over indexed documents |

## Project Structure

```
QueryNest/
├── backend/          # FastAPI server, embeddings, FAISS, Supabase integration
├── frontend/         # Flutter cross-platform client
├── .env.example      # Environment variable template (copy to backend/.env)
├── LICENSE
└── README.md
```

## Contributing

Pull requests are welcome. For larger changes, please open an issue first to discuss the proposed work.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
