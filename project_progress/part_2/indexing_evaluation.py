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
    index = defaultdict(list)
    tf=defaultdict(dict) #why dic in the report
    df=defaultdict(int)
    title_index=defaultdict(str)
    idf=defaultdict(float)
    N=len(docs)

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
    
        tokens = build_terms(combined_text)
        
        counts= collections.Counter(tokens)
        #current_doc_index = {}

        '''for pos, term in enumerate(tokens):
            try:
                current_doc_index[term][1].append(pos)
            except KeyError:
                current_doc_index[term] = [pid, array('I', [pos])]
       
        #norm for TF
        norm=math.sqrt(sum(len(posting[1])**2 for posting in current_doc_index.values())) if current_doc_index else 0.0
        for term, posting in current_doc_index.items():
            tf[term].append(len(posting[1])/norm if norm>0 else 0.0)
            df[term] +=1
            index[term].append(posting)

         '''
        norm=math.sqrt(sum(c**2 for c in counts.values())) or 1.0

        for term,c in counts.items():
            tf_=c/norm
            tf[term][pid]=tf_
            df[term] +=1
            index[term].append((pid,c))


    for term in df:
        idf[term]=math.log(N/df[term]) if df[term]>0 else 0.0

    return index, tf, df, idf, title_index

'''
Search and ranking
'''
def ranking_docs(terms,docs,index,idf,tf,title_index):
    #create a map form pid to tf list index
    pid_idx={}
    for term in terms:
        if term in index:
            for i, (pid,posting) in enumerate(index[term]):
                pid_idx.setdefault(pid,{})[term]=i
    
    vectors_doc=defaultdict(lambda: [0]*len(terms))
    query_vector= [0]*len(terms)
    query_counts=collections.Counter(terms)
    query_norm= np.linalg.norm(list(query_counts.values()))

    for i,term in enumerate(terms):
        if term not in index:
            continue

        query_tf=query_counts[term]/query_norm if query_norm>0 else 0
        query_vector[i]=query_tf*idf.get(term,0.0)

        for pid in docs:
            map_term= pid_idx.get(pid,{})
            if term in map_term:
                vectors_doc[pid][i]=tf[term].get(pid,0.0)*idf.get(term,0.0)

    scores=[[np.dot(v,query_vector),doc] for doc, v in vectors_doc.items()]
    scores.sort(reverse=True)
    return [s[1] for s in scores]

def search_tf_idf(query, index, tf, idf, title_index):
    query_terms=build_terms(query)
    if not query_terms:
        return []

    if query_terms[0] not in index: #start with first term
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
    
    docs=list(docs_set)
   
    return ranking_docs(query_terms,docs,index,idf,tf,title_index)


'''
Main  
'''

if __name__ == "__main__":
    docs=load_processed_docs()
    index,tf,df,idf,title_index=create_index_tfidf(docs)

    queries=[
        "full sleeve black shirt",
        "solid women white polo",
        "print of multicolor neck grey shirt",
        "slim fit men blue jeans",
        "round collar full sleeves t-shirt" #mirar lo de t i sense t
    ]

    for q in queries:
        result=search_tf_idf(q,index,tf,idf,title_index)
        if not result:
            print("No matching documents.")
        else:
            for pid in result[:5]:
                print(f"{pid}: {title_index.get(pid,'[No title]')}")
