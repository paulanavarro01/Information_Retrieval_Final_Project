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

parent_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

#assumption: assuming part_1.dataprocessing has a correct version of build_terms
from part_1.data_processing import build_terms

proc_doc_path='../data/processed_docs.jsonl'
validation_path='../data/validation_labels.csv'


def load_processed_docs(path=proc_doc_path):
    with open(path,'r',encoding='utf-8') as f:
        docs=[json.loads(line)for line in f if line.strip()]
    return docs

'''
Inverted index
'''

def create_index_tfidf(docs):
    index = defaultdict(list)
    tf=defaultdict(list)
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
        
        current_doc_index = {}

        for pos, term in enumerate(tokens):
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
            
    for term in df:
        idf[term]=math.log(N/df[term]) if df[term]>0 else 0.0

    return index, tf, df, idf, title_index

'''
Search  repessar
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
        print(f"Trem:{term}, IDF={idf[term]:.4f}")
        query_tf=query_counts[term]/query_norm if query_norm>0 else 0
        query_vector[i]=query_tf*idf[term] 
        for pid in docs:
            map_term= pid_idx.get(pid,{})
            if term in map_term:
                j=map_term[term]
                vectors_doc[pid][i]=tf[term][j]*idf[term]

    scores=[[np.dot(v,query_vector),doc] for doc, v in vectors_doc.items()]
    scores.sort(reverse=True)
    return [s[1] for s in scores]

def search_tf_idf(query, index, tf, idf, title_index):
    query_terms=build_terms(query)
    print(f"Query:{query}, terms: {query_terms}")
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
            doc_set=set()
            break
    
    print(f"For {query}: {len(docs_set)}")
    if not docs_set:
        return []
    
    docs=list(docs_set)
   
    return ranking_docs(query_terms,docs,index,idf,tf,title_index)

'''
Evaluation
'''

def precision_k(y_true,y_score,k=10):
    order=np.argsort(np.asarray(y_score))[::-1]
    y_true= np.take(np.asarray(y_true),order)
    k=min(k,len(y_true)) #handle if k> len(y_true)
    return np.sum(y_true[:k])/k if k>0 else 0.0

def average_precision(y_true,y_score,k=10):
    order=np.argsort(y_score)[::-1]
    y_true=np.take(np.asarray(y_true),order)
    prec_list=[]
    num_relevant=0
    for i in range(min(k,len(order))):
        if y_true[i]==1:
            num_relevant +=1
            prec_list.append(num_relevant/(i+1))
    return np.sum(prec_list)/num_relevant if num_relevant>0 else 0.0

def recall_k(y_true,y_score,k=10):
    order=np.argsort(np.asarray(y_score))[::-1]
    y_true=np.take(np.asarray(y_true),order)
    relevenat=np.sum(y_true[:k])
    total_rel=np.sum(np.asarray(y_true))
    return relevenat/total_rel if total_rel>0 else 0.0

def f1_k(y_true,y_score,k=10):
    prec=precision_k(y_true,y_score,k)
    rec=recall_k(y_true,y_score,k)
    return (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0

# map_k from lab 2 DON'T KNWO IF WE HAVE TO DO IT
'''
def map_at_k(search_res, k=10):
    Parameters
    ----------
    search_res: search results dataset containing:
        q_id: query id.
        doc_id: document id.
        predicted_relevance: relevance predicted through LightGBM.
        y_true: actual score of the document for the query (ground truth).

    Returns
    -------
    mean average precision @k : float
    avp = []
    for q in search_res["query_id"].unique():  #loop over all query id
        curr_data = search_res[search_res["query_id"] == q]  # select data for current query
        avp.append(avg_precision_at_k(np.array(curr_data["is_relevant"]), np.array(curr_data["predicted_relevance"]),
                                      k))  #append average precision for current query
    return np.sum(avp) / len(avp), avp  # return mean average precision
'''

def rr_k(y_true,y_score,k=10):
    order=np.argsort(np.asarray(y_score))[::-1]
    y_true= np.take(np.asarray(y_true),order)[:k]
    if np.sum(y_true)==0: 
        return 0
    return 1/(np.argmax(y_true)+1)

def dcg_k(y_true,y_score,k=10):
    order=np.argsort(y_score)[::-1]
    y_true=np.take(y_true,order[:k])
    gain= 2**y_true-1
    discounts=np.log2(np.arange(len(y_true))+2) #+2 is added because log2(1) is 0
    return np.sum(gain/discounts)

def ndcg_k(y_true,y_score,k=10):
    dcg= dcg_k(y_true,y_score,k)
    idcg=dcg_k(y_true,y_true,k)
    if not idcg:
        return 0
    return round(dcg/idcg,4)


'''
Main  LOOK AT IT, DON'T UNDERSTAND QUITE , s'ha de treure lo dels debug pk tenia algun errors i volia mirar d'on venien
'''

if __name__ == "__main__":
    docs=load_processed_docs()
    

    index,tf,df,idf,title_index=create_index_tfidf(docs)
    
    '''Loading validation'''
    labels_df=pd.read_csv(validation_path)
    labels_df['pid'] = labels_df['pid'].astype(str).str.strip()
    validation_pids = set(labels_df['pid'])
    
    
    label_groups=defaultdict(dict)
    for _,r in labels_df.iterrows():
        label_groups[str(r['query_id'])][str(r['pid'])]=int(r['labels'])
    
    queries=[
        "women cotton sweatshirt",
        "slim men blue jeans",
        "super skinny women blue jeans",
        "full sleeve printed women sweatshirt",
        "graphic print women sweatshirt"
    ]

    query_map={
        "women cotton sweatshirt":"1",
        "slim men blue jeans":"2",
        "super skinny women blue jeans":"1",
        "full sleeve printed women sweatshirt":"2",
        "graphic print women sweatshirt":"1"
    }
  
    for q in queries:
        q_id=query_map[q]
        print(f"\nQuery: {q}, ID: {q_id}")
        

        rank_docs=search_tf_idf(q,index,tf,idf,title_index)
        
        valid_rank=[pid for pid in rank_docs if pid in validation_pids]
        y_true=np.array([label_groups[q_id].get(pid,0) for pid in valid_rank])
            
        y_score=np.arange(len(y_true),0,-1)
        
        # Debug: show top 10 PIDs and labels (validation only)
        top_10_pids = valid_rank[:10]
        top_10_labels = [label_groups[q_id].get(pid, 0) for pid in top_10_pids]
        print(f"DEBUG: Top 10 PIDs: {top_10_pids}")
        print(f"DEBUG: Top 10 Labels: {top_10_labels}")

        
        
        if len(y_true)>0:
            print(f"Query:{q}")
            print(f"P@10:{precision_k(y_true,y_score,10):.3f}")
            print(f"AP@10:{average_precision(y_true,y_score,10):.3f}")
            print(f"R@10:{recall_k(y_true,y_score,10):.3f}")
            print(f"F1@10:{f1_k(y_true,y_score,10):.3f}")
            print(f"RR@10:{rr_k(y_true,y_score,10):.3f}")
            print(f"NDCG@10:{ndcg_k(y_true,y_score,10):.3f}")
        else:
            print(f"Query: {q} - No relevant documents found")