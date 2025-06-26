# Document Chunk Similarity Analyzer

This project analyzes documents separated by `-----` delimiters, chunks them into smaller pieces, generates embeddings, and finds similar content across different documents.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python main.py
```

With custom parameters:
```bash
python main.py --file data.md --chunk-size 200 --threshold 0.8 --output results.json
```

## Parameters

- `--file`: Input markdown file (default: data.md)
- `--chunk-size`: Maximum chunk size in characters (default: 300)
- `--overlap`: Overlap between chunks (default: 50)  
- `--threshold`: Similarity threshold 0.0-1.0 (default: 0.7)
- `--output`: Output JSON file (default: similarity_results.json)
- `--model`: Sentence transformer model (default: all-MiniLM-L6-v2)

## Output

The tool generates a JSON file containing:
- `similar_pairs`: Array of chunk pairs with similarity scores
- `groups`: Grouped similar chunks

## Example

Your data.md file contains policy documents. The tool will:
1. Parse 30 documents from your file
2. Create ~150+ chunks
3. Generate embeddings for semantic similarity
4. Find chunk pairs above the similarity threshold
5. Group related chunks together

Similar content will be identified across different policy versions, showing which sections have similar information.