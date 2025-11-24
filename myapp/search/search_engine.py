import random
import numpy as np

from myapp.search.objects import Document
from project_progress.part_2.indexing_evaluation import (
    load_processed_docs,
    create_index_tfidf,
    search_tf_idf
)

docs = load_processed_docs()                          # List of dicts
index, tf, df, idf, title_index = create_index_tfidf(docs)

# Build a PID → full document lookup (dictionary)
full_doc_lookup = {doc["pid"]: doc for doc in docs}


def tfidf_search(corpus: dict, search_query, search_id, num_results=20):
    """
    Executes your TF-IDF search and returns Document objects for the UI.
    """

    ranked_pids = search_tf_idf(search_query, index, tf, idf, title_index)

    results = []

    for pid in ranked_pids[:num_results]:

        raw = corpus.get(pid)                    # This is a Document object
        proc = full_doc_lookup.get(pid, {})      # This is the dict from processed docs

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



class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        results = tfidf_search(corpus, search_query, search_id)  # replace with call to search algorithm

        # results = search_in_corpus(search_query)
        return results
