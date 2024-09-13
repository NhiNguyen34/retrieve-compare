from rank_bm25 import BM25Okapi
import numpy as np
from src.common import load_corpus, process_corpus, ParagraphEmbedder, load_queries, parse_args
import pandas as pd

class RetrievalBM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])

    def retrieve(self, query, top_k=5):
        scores = self.bm25.get_scores(query.split())
        top_n = np.argsort(scores)[::-1][:top_k]
        return top_n, scores[top_n]

def main():
    args = parse_args()
    
    corpus = load_corpus(args.corpus_path)
    paragraph_embedder = ParagraphEmbedder()
    
    bm25_retriever = RetrievalBM25(corpus)
    
    queries, data = load_queries(args.queries_path)
    
    results = []
        
    for idx in range(len(data)):
        query = data.iloc[idx]['question']
        processed_query = paragraph_embedder.preprocess(query)[0]
        top_n, scores = bm25_retriever.retrieve(processed_query, top_k=args.top_k)
        context = ' '.join([bm25_retriever.corpus[i] for i in top_n])
        data.at[idx, 'context'] = context # assigns context to the context column
        
    data.to_csv(args.output, index=False)
        

if __name__ == "__main__":
    main()