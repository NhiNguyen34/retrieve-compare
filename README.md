# Information Retrieval System

An information retrieval system comparing the performance of BM25, Cosine Similarity, and a combined approach.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/NhiNguyen34/retrieve-compare.git
   cd retrieve-compare
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `common.py`: Contains common functions and classes.
- `bm25_retrieval.py`: Implements the BM25 retrieval method.
- `cosine_retrieval.py`: Implements the Cosine Similarity retrieval method.
- `combined_retrieval.py`: Implements the combined BM25 and Cosine Similarity method.

## Usage

### BM25 Retrieval
```
python bm25_retrieval.py --corpus_path path/to/corpus.txt --queries_path path/to/queries.txt --top_k 10 --output bm25_results.csv
```

### Cosine Similarity Retrieval
```
python cosine_retrieval.py --corpus_path path/to/corpus.txt --queries_path path/to/queries.txt --top_k 10 --output cosine_results.csv
```

### Combined Retrieval 
```
python combined_retrieval.py --corpus_path path/to/corpus.txt --queries_path path/to/queries.txt --top_k 10 --output combined_results.csv
```


### Parameters

- `--corpus_path`: Path to the corpus file (required).
- `--queries_path`: Path to the queries file (required).
- `--top_k`: Number of top results to retrieve (default: 5).
- `--output`: Name of the output CSV file (default: results.csv).

## Data Format

### Corpus

The corpus file should contain one paragraph of text per line.

### Queries

The queries file should contain one query per line.

## Results

Results will be saved in a CSV file with the following columns:
- `query`: The original query.
- `results`: A list of search results, including the text and score.

## Contributing

Contributions are welcome. Please open an issue to discuss major changes before making them.

## License

[MIT](https://choosealicense.com/licenses/mit/)
