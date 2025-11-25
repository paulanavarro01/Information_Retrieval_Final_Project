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
# Dummy price table (fill with actual data)
# -------------------------
price_table = {
    doc["pid"]: random.randint(10, 500) for doc in docs  # Random prices for testing
}

def get_price(pid):
    """Return price for a document or 0 if unavailable."""
    return price_table.get(pid, 0)

# -------------------------
# TF-IDF search (baseline)
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
# BM25 ranking
# -------------------------
def BM25(query_terms, docs, index, df, title_index, k=1.5, b=0.75):
    N = len(title_index)
    doc_length = defaultdict(int)
    for term, postings in index.items():
        for pid, count in postings:
            doc_length[pid] += count
    avg_dl = sum(doc_length.values()) / len(doc_length) if doc_length else 0

    scores = defaultdict(float)
    for term in query_terms:
        df_term = df.get(term, 0)
        idf_score = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)  # smoothed IDF
        postings = dict(index.get(term, []))
        for pid in docs:
            tf = postings.get(pid, 0)
            if tf == 0: continue
            ld = doc_length.get(pid, avg_dl)
            numerator = tf * (k + 1)
            denominator = tf + k * (1 - b + b * (ld / avg_dl))
            scores[pid] += idf_score * (numerator / denominator)
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked_docs]

def bm25_search(corpus: dict, search_query, search_id, num_results=20):
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
# BM25 + Price Boost (Best Offer mode)
# -------------------------
def price_boost_score(query_terms, docs, index, df, title_index, k=1.5, b=0.75, price_weight=50.0):
    """
    BM25 with additional price boost for cheaper items.
    Increased price_weight to make the effect more visible.
    """
    N = len(title_index)
    doc_length = defaultdict(int)
    for term, postings in index.items():
        for pid, count in postings:
            doc_length[pid] += count
    avg_dl = sum(doc_length.values()) / len(doc_length) if doc_length else 0

    max_price = max(price_table.values()) if price_table else 1
    scores = defaultdict(float)

    for term in query_terms:
        df_term = df.get(term, 0)
        idf_score = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)
        postings = dict(index.get(term, []))
        for pid in docs:
            tf = postings.get(pid, 0)
            if tf == 0: 
                continue
            ld = doc_length.get(pid, avg_dl)
            bm_score = idf_score * ((tf * (k + 1)) / (tf + k * (1 - b + b * (ld / avg_dl))))

            # --- Stronger price boost: logarithmic scaling ---
            price = get_price(pid)
            price_boost = price_weight * (1 - (math.log10(price + 1) / math.log10(max_price + 1)))
            scores[pid] += bm_score + price_boost

            # --- DEBUG LOGGING ---
            print(f"PID: {pid}, BM25: {bm_score:.2f}, Price: {price}, PriceBoost: {price_boost:.2f}, TotalScore: {scores[pid]:.2f}")

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked_docs]

def price_boost_search(corpus: dict, search_query, search_id, num_results=20, k=1.5, b=0.75, price_weight=50.0):
    query_terms = search_query.lower().split()
    all_pids = list(corpus.keys())
    ranked_pids = price_boost_score(query_terms, all_pids, index, df, title_index, k, b, price_weight)

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
