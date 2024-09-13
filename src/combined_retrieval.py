import numpy as np
from src.common import load_corpus, process_corpus, ParagraphEmbedder, load_queries, parse_args
from src.bm25_retrieval import RetrievalBM25
from src.cosine_retrieval import RetrievalCosine
import pandas as pd

def combine_scores(bm25_scores, cosine_scores, alpha=0.5):
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)
    cosine_norm = (cosine_scores - np.min(cosine_scores)) / (np.max(cosine_scores) - np.min(cosine_scores) + 1e-8)
    return alpha * bm25_norm + (1 - alpha) * cosine_norm

def main():
    args = parse_args()
    
    corpus = load_corpus(args.corpus_path)
    paragraph_embedder = ParagraphEmbedder()
    corpus_embeddings = process_corpus(corpus, use_semantic_chunking=args.use_semantic_chunking)
    
    bm25_retriever = RetrievalBM25(corpus)
    cosine_retriever = RetrievalCosine(corpus, corpus_embeddings)
    
    queries, data = load_queries(args.queries_path)
    
    results = []
    
    # for idx in range(len(data)):
    #     query = data.iloc[idx]['question']
    #     processed_query = paragraph_embedder.preprocess(query)[0]
    #     query_embedding = paragraph_embedder.get_paragraph_embedding(processed_query, use_semantic_chunking=args.use_semantic_chunking)
        
    #     bm25_scores = bm25_retriever.bm25.get_scores(processed_query.split())
    #     _, cosine_scores = cosine_retriever.retrieve(query_embedding, top_k=len(corpus))
        
    #     combined_scores = combine_scores(bm25_scores, cosine_scores)
    #     top_n = np.argsort(combined_scores)[::-1][:args.top_k]
    #     data.at[idx, 'context'] = ' '.join([corpus[i] for i in top_n])
    # data.to_csv(args.output, index=False)
    
    for idx in range(len(data)):
        query = data.iloc[idx]['question']
        processed_query = paragraph_embedder.preprocess(query)[0]
        
        # Step 1: Retrieve using BM25
        bm25_scores = bm25_retriever.bm25.get_scores(processed_query.split())
        bm25_top_n = np.argsort(bm25_scores)[::-1][:args.top_k]
        bm25_context = [corpus[i] for i in bm25_top_n]
        
        # Step 2: Use Cosine similarity on BM25 context
        bm25_context_embeddings = [corpus_embeddings[i] for i in bm25_top_n]
        query_embedding = paragraph_embedder.get_paragraph_embedding(processed_query, use_semantic_chunking=args.use_semantic_chunking)
        cosine_scores = cosine_retriever.compute_cosine_similarity(query_embedding, bm25_context_embeddings)
        
        # Combine BM25 context based on Cosine scores
        combined_top_n = np.argsort(cosine_scores)[::-1][:args.top_k]
        data.at[idx, 'context'] = ' '.join([bm25_context[i] for i in combined_top_n])
    
    data.to_csv(args.output, index=False)
    

if __name__ == "__main__":
    main()