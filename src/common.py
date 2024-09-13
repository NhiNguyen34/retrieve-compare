import torch
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Information Retrieval System")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--output", type=str, default="results.csv", help="Output file name")
    parser.add_argument("--use_semantic_chunking", action="store_true", help="Use semantic chunking")
    return parser.parse_args()

def preprocessing_data(sample):
    punct = set('!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~')
    sample = ''.join(ch for ch in sample if ch not in punct)
    sample = re.sub(r"\s+", " ", sample)
    return sample.strip().lower()

class ParagraphEmbedder:
    def __init__(self, model_name='vinai/phobert-base', max_chunk_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_chunk_length = max_chunk_length

    def preprocess(self, paragraph):
        return [preprocessing_data(sent) for sent in paragraph.split('.') if sent.strip()]

    def split_and_chunk(self, sentences):
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(self.tokenizer.encode(current_chunk + sentence)) > self.max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def semantic_chunking(self, sentences, similarity_threshold=0.7):
        chunks = []
        current_chunk = []
        chunk_embedding = None

        for sentence in sentences:
            sentence_embedding = self.get_sentence_embedding(sentence)
            
            if not current_chunk:
                current_chunk.append(sentence)
                chunk_embedding = sentence_embedding
            else:
                similarity = cosine_similarity([chunk_embedding], [sentence_embedding])[0][0]
                if similarity >= similarity_threshold and len(self.tokenizer.encode(' '.join(current_chunk + [sentence]))) <= self.max_chunk_length:
                    current_chunk.append(sentence)
                    chunk_embedding = np.mean([chunk_embedding, sentence_embedding], axis=0)
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    chunk_embedding = sentence_embedding

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def get_sentence_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=self.max_chunk_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def embed_chunks(self, chunks):
        embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=self.max_chunk_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return np.array(embeddings)

    def pool_embeddings(self, embeddings):
        return np.mean(embeddings, axis=0)

    def get_paragraph_embedding(self, paragraph, use_semantic_chunking=False):
        sentences = self.preprocess(paragraph)
        if use_semantic_chunking:
            chunks = self.semantic_chunking(sentences)
        else:
            chunks = self.split_and_chunk(sentences)
        chunk_embeddings = self.embed_chunks(chunks)
        paragraph_embedding = self.pool_embeddings(chunk_embeddings)
        return paragraph_embedding

def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split("\n\n")

def process_corpus(corpus, use_semantic_chunking=False):
    paragraph_embedder = ParagraphEmbedder()
    return [paragraph_embedder.get_paragraph_embedding(para, use_semantic_chunking) for para in corpus]

# def load_queries(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return [q.strip() for q in f.readlines()]

def load_queries(file_path):
    data = pd.read_csv(file_path)
    queries = data['question'].tolist()
    return queries, data