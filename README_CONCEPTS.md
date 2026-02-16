# Project Concepts (LangChain + Local RAG)

This document explains the key concepts and how they are used in this project.

## Retrieval-Augmented Generation (RAG)
RAG combines two steps:
1) **Retrieve** relevant chunks from your local knowledge base (ChromaDB).
2) **Generate** an answer grounded in those chunks.

In this project:
- Retrieval is done with Chroma similarity search.
- Generation is done by a local Llama model via Ollama or llama.cpp.

## Chunking
Large documents are split into smaller, overlapping pieces so retrieval can find the best parts.
- We use `RecursiveCharacterTextSplitter`.
- `CHUNK_SIZE` and `CHUNK_OVERLAP` control size and overlap.

## Embeddings
Embeddings turn text into vectors so we can do similarity search.
- Primary: Ollama embeddings (e.g., `nomic-embed-text`).
- Fallback: local sentence-transformers via `langchain-huggingface`.

## Vector Store (ChromaDB)
Chroma stores vectors + metadata on disk.
- Persistent storage in `backend/.chroma`.
- Metadata: source, page, chunk_id, created_at, type, question (for FAQ).

## Retrieval
At chat time, the top `k` chunks are selected and passed into the prompt.
- Optional `source` filter allows searching a specific file.

## Grounded Prompt + Citations
We build a prompt that includes:
- The user question
- Retrieved context
- A list of citations (source/page/chunk_id/score)

The model must return:
- `answer`
- `citations`
- `followups`

If the context is insufficient, the answer is:
`I don't know based on the provided documents.`

## LangChain workflow (small)

1) **Ingest**: Load files → chunk → embed → store in Chroma.
2) **Retrieve**: For a question, fetch top‑k similar chunks (optionally filtered by source).
3) **Route / Tools**: A small LCEL runnable decides if we should call a tool or do normal RAG.
4) **Generate**: Build a grounded prompt with context + citations → local LLM returns JSON.
5) **Validate**: If JSON is malformed or missing citations, retry once with a fix‑format prompt.

## LangChain Runnables (LCEL)
We use LCEL for:
- Routing decisions (tool use vs. normal chat)
- Tool execution

This avoids deprecated APIs and keeps the chain small and deterministic.

## Diagram

```
User Question
     |
     v
  Retriever (Chroma + embeddings)
     |
     +--> LCEL Router ----> Tool? ----> Tool Output
     |            |
     |            v
     |        Normal RAG
     v
 Grounded Prompt (context + citations)
     |
     v
 Local LLM (Ollama / llama.cpp)
     |
     v
 JSON Response (answer, citations, followups)
     |
     v
 Validation + Retry (if needed)
```

## Tool Calling
The assistant can call a limited set of tools:
- `list_sources()` for stats
- `get_context(query, top_k)` for retrieved chunks
- `summarize_chunk(chunk_id)` to summarize one chunk

## Local LLM Providers
Two adapters are available:
- **Ollama**: `http://localhost:11434`
- **llama.cpp server**: `http://localhost:8080`

Selection is done via env vars:
- `LLM_PROVIDER=ollama|llamacpp`
- `LLM_MODEL=<model name>`

## Validation + Reliability
- JSON output is strictly enforced.
- A retry is done if the response is malformed or missing citations.
- Each run is logged with request_id, chunk IDs, model, latency.
