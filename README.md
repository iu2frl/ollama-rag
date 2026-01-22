# RAG with Ollama (qwen3-embedding + aya-expanse)

This repository demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using:

- Ollama for embeddings and LLM inference
- LangChain-style tooling for document loading, splitting and vector storage


The main script is `test.py`.

## Quick overview

1. Load a set of web pages and convert them into documents.
1. Split documents into chunks.
1. Compute embeddings with the local Ollama embedding model `qwen3-embedding`.
1. Store embeddings in a local SKLearn vector store and persist them to disk (`data/embeddings.json`).
1. Use a retriever to fetch the most-relevant chunks for a query and pass them to the `aya-expanse` model running under Ollama to produce a concise answer.

## Prerequisites

- [Ollama download](https://ollama.com/download)
- Python 3.12+ and a virtual environment
- Models pulled into Ollama:

```bash
ollama pull qwen3-embedding
ollama pull aya-expanse
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

1. Install the Python requirements (there should be a `requirements.txt` in the repo):

```bash
pip install -r requirements.txt
```

1. Make sure the Ollama server is running locally:

```bash
ollama serve
```

## Run

Start the script:

```bash
python test.py
```

On first run the script will build embeddings and persist them to `data/embeddings.json`. On subsequent runs it will load that file and skip re-embedding.

## Customization

- Change which web pages are loaded by editing the `urls` list in `test.py`.
- Change the embedding or LLM model names by editing the calls to `OllamaEmbeddings(model=...)` and `ChatOllama(model=...)` in `test.py`.
- Change the persistence location by editing the `persist_path` variable (default: `data/embeddings.json`).
- Change the serializer type by modifying the SKLearnVectorStore constructor (supports `json`, `bson`, `parquet`).

## Troubleshooting

- If you see errors about a missing model, ensure the model is pulled via `ollama pull <model-name>` and that `ollama serve` is running.
- If you want to force a rebuild of embeddings, delete `data/embeddings.json` and re-run the script.
- Check logs printed to stdout for helpful messages (the script uses structured logging).

## License

This project is released under the MIT License.
