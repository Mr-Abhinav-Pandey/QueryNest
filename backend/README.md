# QueryNest Backend

FastAPI server for semantic PDF search using Sentence Transformers, FAISS, and Supabase.

For project overview, environment setup, and full-stack instructions, see the [root README](../README.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
cp ../.env.example .env          # then edit .env with your Supabase credentials
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
