import json
import math
import os
import re
import collections
from collections import defaultdict
from array import array
import numpy as np
import pandas as pd
import sys



#parent_dir=os.path.dirname(os.path.abspath(__file__))
#sys.path.append(parent_dir)
project_root= os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

#assumption: assuming part_1.dataprocessing has a correct version of build_terms
from part_1.data_processing import build_terms

#Import the processed and validation csv from the Data folder
proc_doc_path='../data/processed_docs.jsonl'


def load_processed_docs(path=proc_doc_path):
    '''
    Loads the processed product documents stored in JSONL format.
    Each line is one JSON line representing one product.
    '''
    with open(path,'r',encoding='utf-8') as f:
        docs=[json.loads(line)for line in f if line.strip()]
    return docs

'''
Inverted index
'''
 
def create_index_tfidf(docs):
    """
    Build an inverted index and computes TF, IDF and DF.
    
    The function returns:
        index: {term: [(pid, count), ...]}
        tf: {term: {pid: normalized_tf, ...}}
        idf: {term: inverse_document_frequency}
        df: {term: document_frequency}
        tittle_index: {pid: product title}
    """
    
    #create dictionaries
    index = defaultdict(list)
    tf=defaultdict(dict) #why dic in the report
    df=defaultdict(int)
    title_index=defaultdict(str)
    idf=defaultdict(float)
    
    N=len(docs)
    
    
    #loop through each product document
    for doc in docs:
        pid = doc.get("pid")
        title_index[pid]=doc.get("title_clean","")
        
        # Combine relevant text fields
        combined_text = " ".join([
            doc.get("title_clean", ""),
            doc.get("description_clean", ""),
            doc.get("brand", ""),
            doc.get("category", ""),
            doc.get("sub_category", ""),
            doc.get("seller", ""),
            doc.get("product_details_clean", "")
        ])

        #tokenize and normalize each doc
        tokens = build_terms(combined_text)
        
        #count occurrences of each term within the document
        counts= collections.Counter(tokens)

        #compute normalization factor for TF
        norm=math.sqrt(sum(c**2 for c in counts.values())) or 1.0

        #fill TF and index
        for term,c in counts.items():
            tf_=c/norm
            tf[term][pid]=tf_
            df[term] +=1
            index[term].append((pid,c))

    #compute IDF for each term
    for term in df:
        idf[term]=math.log(N/df[term]) if df[term]>0 else 0.0

    return index, tf, df, idf, title_index

'''
Search and ranking
'''

def ranking_docs(terms,docs,index,idf,tf,title_index):
    """
    Compute a ranking score for each document using the TF-IDF 
    cosine similarity between the query and document vectors
    
    """
    
    #create a map form pid to tf list index
    pid_idx={}
    for term in terms:
        if term in index:
            for i, (pid,posting) in enumerate(index[term]):
                pid_idx.setdefault(pid,{})[term]=i
                
    #initialize query and documnent vectors
    vectors_doc=defaultdict(lambda: [0]*len(terms))
    query_vector= [0]*len(terms)
    
    #compute query term frequencies
    query_counts=collections.Counter(terms)
    query_norm= np.linalg.norm(list(query_counts.values()))

    #build query vector and compute document vectors
    for i,term in enumerate(terms):
        if term not in index:
            continue
        
        #compute query TF-IDF
        query_tf=query_counts[term]/query_norm if query_norm>0 else 0
        query_vector[i]=query_tf*idf.get(term,0.0)

        # compute document TF-IDF for each doc containing the term
        for pid in docs:
            map_term= pid_idx.get(pid,{})
            if term in map_term:
                vectors_doc[pid][i]=tf[term].get(pid,0.0)*idf.get(term,0.0)

    #compute cosine similarity between query and document vectors
    scores=[[np.dot(v,query_vector),doc] for doc, v in vectors_doc.items()]
    scores.sort(reverse=True)
    return [s[1] for s in scores]


def search_tf_idf(query, index, tf, idf, title_index):
    """
    Executes a search query usinf AND logic. 
    Only documents containing all query terms are considered.
    Results are ranqued by TF-IDF cosine similarity.
    
    """
    query_terms=build_terms(query)
    if not query_terms:
        return []

    #start with first query term
    if query_terms[0] not in index: 
        return []

    
    docs_set= set(posting[0]for posting in index[query_terms[0]])

    #only keeping documents that are present in all term's postings
    for term in query_terms[1:]:
        if term in index:
            term_docs= [posting[0] for posting in index[term]]
            docs_set &= set(term_docs)
        else:
            docs_set=set()
            break
    
    if not docs_set:
        return []
    
    #rank final set of documents
    docs=list(docs_set)
   
    return ranking_docs(query_terms,docs,index,idf,tf,title_index)


'''
Main code
'''

if __name__ == "__main__":
    
    #load preprocessed docs
    docs=load_processed_docs()
    
    #build inverted index and compute TF-IDF values
    index,tf,df,idf,title_index=create_index_tfidf(docs)

    
    #test queries
    queries=[
        "full sleeve black shirt",
        "solid women white polo",
        "print of multicolor neck grey shirt",
        "slim fit men blue jeans",
        "round collar full sleeves t-shirt" #mirar lo de t i sense t
    ]
    
    #run and display top results for each query
    for q in queries:
        result=search_tf_idf(q,index,tf,idf,title_index)
        if not result:
            print("No matching documents.")
        else:
            for pid in result[:5]:
                print(f"{pid}: {title_index.get(pid,'[No title]')}")
