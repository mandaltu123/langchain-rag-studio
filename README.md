# RAG Studio (LangChain)

**A modular, LangChain-powered RAG backend** with swappable embedding models, pluggable vector stores, and a clean FastAPI interface. Built as a production-ready RAG foundation — swap in any LLM or vector store without rewriting the pipeline.

## What It Does

- Ingests documents → chunks → embeds → stores in ChromaDB
- Handles chat queries via LangChain retrieval chains with conversation memory
- Supports swappable LLMs: OpenAI or local Ollama (config-driven)
- Exposes retrieval and ingest as FastAPI endpoints
- Tool-ready: retrieval is exposed as a LangChain `Tool` for use in agents
- Includes a test suite for end-to-end RAG validation

## Pipeline

```
Documents (text/PDF)
        │
        ▼
  Chunking (configurable size + overlap)
        │
        ▼
  Embeddings (OpenAI or local)
        │
        ▼
  ChromaDB vector store
        │
        ▼
  LangChain RetrievalChain (with chat history)
        │
        ▼
  LLM response (OpenAI GPT / Ollama)
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env

uvicorn backend.app.main:app --reload
# API docs at http://localhost:8000/docs
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/ingest` | Ingest documents into the vector store |
| POST | `/chat` | Query the RAG pipeline |
| GET | `/health` | Health check |

## Tech Stack

- **Orchestration**: LangChain (chains, retrievers, memory)
- **Embedding**: OpenAI `text-embedding-3-small` or Ollama local
- **Vector Store**: ChromaDB
- **LLM**: OpenAI GPT-4o-mini or Ollama LLaMA
- **API**: FastAPI + Uvicorn

## Project Structure

```
LocalRAGStudio-Langchain/
├── backend/app/
│   ├── main.py           # FastAPI app
│   ├── rag/
│   │   ├── chains.py     # LangChain retrieval chains
│   │   ├── ingest.py     # Document ingestion pipeline
│   │   ├── store.py      # ChromaDB vector store wrapper
│   │   ├── retrieve.py   # Retrieval with top-k and filters
│   │   ├── embeddings.py # Embedding model abstraction
│   │   ├── llm.py        # LLM abstraction (OpenAI / Ollama)
│   │   └── tools.py      # LangChain Tool wrappers
│   └── core/config.py
├── tests/
└── requirements.txt
```
