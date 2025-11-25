import random
import math
from collections import defaultdict

from myapp.search.objects import Document
from project_progress.part_2.indexing_evaluation import (
    load_processed_docs,
    create_index_tfidf,
    search_tf_idf
)

# -------------------------
# Load documents and create TF-IDF index
# -------------------------
docs = load_processed_docs()                          # List of dicts
index, tf, df, idf, title_index = create_index_tfidf(docs)

# Build a PID → full document lookup (dictionary)
full_doc_lookup = {doc["pid"]: doc for doc in docs}

# -------------------------
# TF-IDF search (already working)
# -------------------------
def tfidf_search(corpus: dict, search_query, search_id, num_results=20):
    ranked_pids = search_tf_idf(search_query, index, tf, idf, title_index)

    results = []
    for pid in ranked_pids[:num_results]:
        raw = corpus.get(pid)
        proc = full_doc_lookup.get(pid, {})

        if raw is None:
            continue

        title = raw.title or proc.get("title_clean") or "[No Title]"
        description = raw.description or proc.get("description_clean") or ""

        results.append(
            Document(
                pid=pid,
                title=title,
                description=description,
                url=f"doc_details?pid={pid}&search_id={search_id}",
                ranking=random.random()
            )
        )

    return results

# -------------------------
# BM25 implementation
# -------------------------
def BM25(query_terms, docs, index, df, title_index, k=1.5, b=0.75):
    """
    Standard BM25 ranking function.
    Returns a list of ranked document PIDs.
    """
    N = len(title_index)

    # Compute document lengths
    doc_length = defaultdict(int)
    for term, postings in index.items():
        for pid, count in postings:
            doc_length[pid] += count
    avg_dl = sum(doc_length.values()) / len(doc_length) if doc_length else 0

    scores = defaultdict(float)

    for term in query_terms:
        df_term = df.get(term, 0)
        idf_score = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)  # smoothed IDF

        postings = dict(index.get(term, []))  # convert list of tuples to dict {pid: tf}

        for pid in docs:
            tf = postings.get(pid, 0)
            if tf == 0:
                continue

            ld = doc_length.get(pid, avg_dl)
            numerator = tf * (k + 1)
            denominator = tf + k * (1 - b + b * (ld / avg_dl))
            scores[pid] += idf_score * (numerator / denominator)

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, score in ranked_docs]

# -------------------------
# BM25 search wrapper
# -------------------------
def bm25_search(corpus: dict, search_query, search_id, num_results=20):
    """
    Performs BM25 ranking and returns Document objects for UI.
    """
    # Simple query tokenizer
    query_terms = search_query.lower().split()

    all_pids = list(corpus.keys())

    ranked_pids = BM25(query_terms, all_pids, index, df, title_index)

    results = []
    for pid in ranked_pids[:num_results]:
        raw = corpus.get(pid)
        proc = full_doc_lookup.get(pid, {})

        if raw is None:
            continue

        title = raw.title or proc.get("title_clean") or "[No Title]"
        description = raw.description or proc.get("description_clean") or ""

        results.append(
            Document(
                pid=pid,
                title=title,
                description=description,
                url=f"doc_details?pid={pid}&search_id={search_id}",
                ranking=random.random()
            )
        )

    return results


# -------------------------
# BM25 with Price Boost
# -------------------------
price_table = {}  # You can fill this with pid -> price mapping

def get_price(pid):
    """Return price for a document or 0 if unavailable."""
    return price_table.get(pid, 0)

def our_score(query_terms, docs, index, df, title_index, k=1.5, b=0.75):
    N = len(title_index)
    doc_length = defaultdict(int)
    for term, postings in index.items():
        for pid, count in postings:
            doc_length[pid] += count
    avg_dl = sum(doc_length.values()) / len(doc_length) if doc_length else 0

    scores = defaultdict(float)
    for term in query_terms:
        df_term = df.get(term, 0)
        idf_score = math.log((N + 0.5) / (df_term + 0.5))
        postings = dict(index.get(term, []))
        for pid in docs:
            tf = postings.get(pid, 0)
            ld = doc_length.get(pid, avg_dl)
            denominator = tf + k * ((1 - b) + b * (ld / avg_dl))
            bm_score = idf_score * ((tf * (k + 1)) / denominator)

            # Price boosting
            price = get_price(pid)
            price_boost = 1 + (1 + math.log10(price + 1)) if price > 0 else 1.0
            scores[pid] += bm_score * price_boost

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked_docs]

def price_boost_search(corpus: dict, search_query, search_id, num_results=20, k=1.5, b=0.75):
    query_terms = search_query.lower().split()
    all_pids = list(corpus.keys())
    ranked_pids = our_score(query_terms, all_pids, index, df, title_index, k, b)

    results = []
    for pid in ranked_pids[:num_results]:
        raw = corpus.get(pid)
        proc = full_doc_lookup.get(pid, {})
        if raw is None:
            continue
        title = raw.title or proc.get("title_clean") or "[No Title]"
        description = raw.description or proc.get("description_clean") or ""
        results.append(
            Document(
                pid=pid,
                title=title,
                description=description,
                url=f"doc_details?pid={pid}&search_id={search_id}",
                ranking=random.random()
            )
        )
    return results