import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.common import load_corpus, process_corpus, ParagraphEmbedder, load_queries, parse_args
import pandas as pd

class RetrievalCosine:
    def __init__(self, corpus, corpus_embeddings):
        self.corpus = corpus
        self.corpus_embeddings = corpus_embeddings

    def retrieve(self, query_embedding, top_k=5):
        similarities = cosine_similarity([query_embedding], self.corpus_embeddings)[0]
        top_n = np.argsort(similarities)[::-1][:top_k]
        return top_n, similarities[top_n]

def main():
    args = parse_args()
    
    corpus = load_corpus(args.corpus_path)
    paragraph_embedder = ParagraphEmbedder()
    corpus_embeddings = process_corpus(corpus, use_semantic_chunking=args.use_semantic_chunking)
    
    cosine_retriever = RetrievalCosine(corpus, corpus_embeddings)
    
    queries, data = load_queries(args.queries_path)
    
    results = []
    
    for idx in range(len(data)):
        query = data.iloc[idx]['question']
        query_embedding = paragraph_embedder.get_paragraph_embedding(query, use_semantic_chunking=args.use_semantic_chunking)
        top_n, scores = cosine_retriever.retrieve(query_embedding, top_k=args.top_k)
        data.at[idx, 'context'] = ' '.join([corpus[i] for i in top_n])
        
    data.to_csv(args.output, index=False)
    
    # for query in queries:
    #     query_embedding = paragraph_embedder.get_paragraph_embedding(query, use_semantic_chunking=args.use_semantic_chunking)
    #     top_n, scores = cosine_retriever.retrieve(query_embedding, top_k=args.top_k)
    #     results.append({
    #         'query': query,
    #         'cosine_results': [{'text': cosine_retriever.corpus[i], 'score': score} for i, score in zip(top_n, scores)]
    #     })
    
    # df = pd.DataFrame(results)
    # df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()