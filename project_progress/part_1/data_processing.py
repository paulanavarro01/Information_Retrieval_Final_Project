'''''
Part1: This file is used for Text Processing and Data Analysy

'''''

import collections
from collections import defaultdict
import re
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
from array import array

nltk.download("stopwords")

#load data into memory:
docs_path = '../../data/fashion_products_dataset.json'
with open(docs_path) as fp:
    lines = fp.readlines()
lines = [l.strip().replace(' +', ' ') for l in lines]

def build_terms(text):
    '''
    Function used to clean the text before indexing
    
    '''
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def extract_product_details(details_list):
    '''
    This function converts a list of dicts in 'Product details' into a single string
    '''
    if not isinstance(details_list, list):
        return ""
    text_parts = []
    for d in details_list:
        if isinstance(d, dict):
            for k, v in d.items():
                text_parts.append(str(k))
                text_parts.append(str(v))
    return " ".join(text_parts)

def parse_discount(discount_str):
    '''
    Function used to replace the discount field from: "48% off" to 0.48 in order to simplfy the processing of the data.
    
    '''
    if not discount_str:
        return 0.0
    match = re.search(r"(\d+)", discount_str)
    return float(match.group(1)) / 100 if match else 0.0


def parse_float(value):
    """
    Convert strings like '1,499' or '3.9' to float
    Returns None if conversion fails
    """
    if not value:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None
    

def clean_url(url):
    '''
    Function done to clean the URLs since all the urls can be shortened to the "?" and will display the same URL
    '''
    if not isinstance(url, str):
        return ""
    return url.split("?")[0]

def create_index(docs):
    '''
    Function done to create a dictionary of indexes with the words and the documents they appear in
    
    '''

    index = defaultdict(list)
    for doc in docs:
        pid = doc["pid"]
        # Combine relevant text fields
        combined_text = " ".join([
            doc.get("title", ""),
            doc.get("description", ""),
            doc.get("brand", ""),
            doc.get("category", ""),
            doc.get("sub_category", ""),
            doc.get("seller", ""),
            extract_product_details(doc.get("product_details", []))
        ])
    
        tokens = build_terms(combined_text)
        current_doc_index = {}

        for pos, term in enumerate(tokens):
            try:
                current_doc_index[term][1].append(pos)
            except KeyError:
                current_doc_index[term] = [pid, array('I', [pos])]

        for term, posting in current_doc_index.items():
            index[term].append(posting)

    return index


def preprocess_dataset(input_path, output_path_index, output_path_clean):
    """
    Read the dataset, clean text, and build an inverted index.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    cleaned_docs = []
    for doc in docs:
        clean_title = " ".join(build_terms(doc.get("title", "")))
        clean_desc = " ".join(build_terms(doc.get("description", "")))
        clean_details = extract_product_details(doc.get("product_details", []))

        cleaned_docs.append({
            "pid": doc.get("pid"),
            "title_clean": clean_title,
            "description_clean": clean_desc,
            "product_details_clean": clean_details,
            "brand": doc.get("brand", ""),
            "category": doc.get("category", ""),
            "sub_category": doc.get("sub_category", ""),
            "seller": doc.get("seller", ""),
            "out_of_stock": doc.get("out_of_stock", ""),
            "selling_price": parse_float(doc.get("selling_price", "")),
            "discount": parse_discount(doc.get("discount", "")),
            "actual_price": parse_float(doc.get("actual_price", "")),
            "average_rating": parse_float(doc.get("average_rating", "")),
            "url": clean_url(doc.get("url", ""))
        })

    with open(output_path_clean, "w", encoding="utf-8") as f:
        for doc in cleaned_docs:
            f.write(json.dumps(doc) + "\n")

    print(f" Saved cleaned docs to {output_path_clean}")

    index = create_index(docs)
    with open(output_path_index, "w", encoding="utf-8") as f:
        json.dump({term: [[pid, list(pos)] for pid, pos in postings]
                  for term, postings in index.items()}, f)

    print(f" Inverted index saved to {output_path_index}")


if __name__ == "__main__":
    preprocess_dataset(
        input_path="../../data/fashion_products_dataset.json",
        output_path_index="../../data/inverted_index.json",
        output_path_clean="../../data/processed_docs.jsonl"
    )