import time
import statistics
import numpy as np
from functions import semantic_search_instrumented, pdf_texts, load_index_from_db, index
from models import SearchRequest

NUM_RUNS = 100
TOP_K = 5


def percentile(arr, p):
    return float(np.percentile(arr, p))


def compute_stats(arr):
    return {
        "avg": statistics.mean(arr) if arr else 0.0,
        "median": statistics.median(arr) if arr else 0.0,
        "p95": percentile(arr, 95) if arr else 0.0,
        "worst": max(arr) if arr else 0.0,
    }


def main():
    # Ensure index is loaded from DB before benchmarking
    load_index_from_db()
    if len(pdf_texts) == 0:
        print("Warning: `pdf_texts` is empty — no indexed documents found. Bench will still run but results may be trivial.")

    # Report counts after rehydration
    try:
        faiss_ntotal = int(index.ntotal)
    except Exception:
        faiss_ntotal = index.ntotal
    print(f"documents_loaded={len(pdf_texts)}")
    print(f"faiss_ntotal={faiss_ntotal}")
    print(f"len(pdf_texts)={len(pdf_texts)}")

    # Choose a query from indexed content if available
    if len(pdf_texts) > 0:
        sample_query = pdf_texts[0]["content"][:120].strip() or "example query"
    else:
        sample_query = "example query"

    embed_times = []
    faiss_times = []
    assembly_times = []
    total_times = []

    # Warm-up
    _res, _t = semantic_search_instrumented(SearchRequest(query=sample_query, top_k=TOP_K))

    for i in range(NUM_RUNS):
        _, timings = semantic_search_instrumented(SearchRequest(query=sample_query, top_k=TOP_K))
        embed_times.append(timings.get("embedding_time", 0.0))
        faiss_times.append(timings.get("faiss_time", 0.0))
        assembly_times.append(timings.get("assembly_time", 0.0))
        total_times.append(timings.get("total_time", 0.0))

    stats = {
        "embedding": compute_stats(embed_times),
        "faiss": compute_stats(faiss_times),
        "assembly": compute_stats(assembly_times),
        "total": compute_stats(total_times),
    }

    # Print table
    print("\nBenchmark results ({} runs)\n".format(NUM_RUNS))
    print("Metric       | avg (s)  | median (s) | p95 (s)  | worst (s)")
    print("----------------------------------------------------------")
    for key in ["embedding", "faiss", "assembly", "total"]:
        s = stats[key]
        print(f"{key.ljust(12)}| {s['avg']:.6f} | {s['median']:.6f}   | {s['p95']:.6f} | {s['worst']:.6f}")
    
    print(f"Documents loaded: {len(pdf_texts)}")
    print(f"FAISS vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
