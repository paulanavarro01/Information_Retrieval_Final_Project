import random
import numpy as np

from myapp.search.algorithms import (tfidf_search, bm25_search, price_boost_search)



class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, corpus, method="bm25"):
        print("Search query:", search_query)
        results = []

        # choose algorithm based on method
        if method == "bm25":
            from myapp.search.algorithms import bm25_search
            results = bm25_search(corpus, search_query, search_id)
        elif method == "price_boost":
            from myapp.search.algorithms import price_boost_search
            results = price_boost_search(corpus, search_query, search_id)
        else:
            from myapp.search.algorithms import tfidf_search
            results = tfidf_search(corpus, search_query, search_id)

        return results