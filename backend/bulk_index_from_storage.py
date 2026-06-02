import argparse
import io
import time
import traceback
from typing import List

from functions import (
    supabase,
    BUCKET_NAME,
    extract_text_from_pdf,
    compute_file_hash,
    model,
    index,
    pdf_texts,
)


def download_object(name: str) -> bytes:
    """Download an object from Supabase storage and return bytes.
    Returns None on failure.
    """
    try:
        resp = supabase.storage.from_(BUCKET_NAME).download(name)
        # supabase-py may return raw bytes or a dict with 'data'
        if resp is None:
            return None
        if isinstance(resp, (bytes, bytearray)):
            return bytes(resp)
        if isinstance(resp, dict) and resp.get("error"):
            return None
        # some clients wrap bytes in {'data': b'...'}
        if isinstance(resp, dict) and resp.get("data"):
            return resp.get("data")
        # fallback: try to return resp directly
        return resp
    except Exception:
        return None


class _FileLike:
    def __init__(self, b: bytes):
        self.file = io.BytesIO(b)


def process_file(name: str, commit: bool) -> (bool, str):
    """Process a single storage object: download, extract, embed, insert, index.
    Returns (success, message)
    """
    try:
        data = download_object(name)
        if not data:
            return False, "download_failed"

        # compute hash
        fh = compute_file_hash(data)

        # check if already present in DB by file_hash or file_path
        q = supabase.table("documents").select("id,file_hash,file_path,filename").eq("file_hash", fh).execute()
        if q.data:
            return False, "already_in_db_by_hash"

        q2 = supabase.table("documents").select("id,file_hash,file_path,filename").eq("file_path", name).execute()
        if q2.data:
            return False, "already_in_db_by_path"

        # If this is a dry-run, avoid downloading/extracting/embedding
        if not commit:
            return True, "dry_run_ok"

        # extract text
        fl = _FileLike(data)
        text = extract_text_from_pdf(fl)
        if not text or not text.strip():
            return False, "no_text_extracted"

        # encode
        emb = model.encode([text])

        if commit:
            payload = {
                "filename": name,
                "file_path": name,
                "content": text,
                "embedding": emb[0].tolist(),
                "file_hash": fh,
            }
            try:
                supabase.table("documents").insert(payload).execute()
            except Exception as e:
                return False, f"db_insert_failed: {e}"

            # add to in-process FAISS index and in-memory list
            try:
                index.add(emb.astype("float32"))
                pdf_texts.append({"filename": name, "content": text, "file_path": name})
            except Exception as e:
                # indexing failed, but DB has the row — leave for startup rehydration
                return False, f"faiss_index_failed: {e}"

            return True, "indexed"
        else:
            return True, "dry_run_ok"

    except Exception as e:
        tb = traceback.format_exc()
        return False, f"exception: {e}\n{tb}"


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(description="Bulk index PDFs from Supabase Storage into DB + FAISS")
    parser.add_argument("--commit", action="store_true", help="Apply changes. Default is dry-run.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = all)")
    parser.add_argument("--sample", type=int, default=0, help="Process only the first N candidates (dry-run helpful)")
    args = parser.parse_args(argv)

    commit = args.commit

    print(f"Listing objects in bucket '{BUCKET_NAME}'...")
    objs = supabase.storage.from_(BUCKET_NAME).list() or []
    pdfs = [o.get("name") for o in objs if o.get("name") and o.get("name").lower().endswith(".pdf")]
    total = len(pdfs)
    print(f"Found {total} pdf objects in storage")

    # load existing DB rows
    docs_resp = supabase.table("documents").select("id,filename,file_path,file_hash").execute()
    docs = docs_resp.data or []
    existing_paths = set((d.get("file_path") or d.get("filename") or "") for d in docs)
    existing_hashes = set(d.get("file_hash") for d in docs if d.get("file_hash"))

    # select candidates
    candidates = [p for p in pdfs if p not in existing_paths]
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if args.sample > 0:
        candidates = candidates[: args.sample]

    print(f"Candidates to process: {len(candidates)} (existing in DB: {len(pdfs) - len(candidates)})")

    processed = 0
    succeeded = 0
    skipped = 0
    failed = 0
    failures = []

    start = time.time()
    for i, name in enumerate(candidates, 1):
        processed += 1
        print(f"[{processed}/{len(candidates)}] Processing: {name}  ", end="", flush=True)
        ok, msg = process_file(name, commit=commit)
        if ok:
            succeeded += 1
            print(f"OK ({msg})")
        else:
            failed += 1
            failures.append({"name": name, "reason": msg})
            print(f"FAILED ({msg})")

    duration = time.time() - start
    print("\nSummary:")
    print(f"  storage_pdfs={total}")
    print(f"  candidates_processed={processed}")
    print(f"  succeeded={succeeded}")
    print(f"  failed={failed}")
    print(f"  duration_s={duration:.2f}")

    if failures:
        print("Failures (sample 20):")
        for f in failures[:20]:
            print(f" - {f['name']}: {f['reason']}")


if __name__ == "__main__":
    main()
