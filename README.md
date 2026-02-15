# LocalRAG Studio (LangChain + FastAPI)

A local RAG project with ChromaDB persistence, local Llama providers (Ollama or llama.cpp), and a minimal FastAPI UI.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use pyenv:

```bash
pyenv install 3.11.8
pyenv local 3.11.8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app.main:app --reload --app-dir backend --port 8010
```

Open UI: `http://localhost:8010/`
Swagger (optional): `http://localhost:8010/docs`

## Run Details

- Default port in examples: `8010` (change with `--port` if needed).
- `.env` is auto-loaded at startup (see `.env` for local overrides).
- If port is in use, pick another (e.g., `--port 8011`).

## Local LLM setup

### Ollama

```bash
# Install Ollama, then
ollama pull llama3.1
ollama pull nomic-embed-text
```

Env (optional):

```bash
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:latest
export OLLAMA_TIMEOUT=30
export OLLAMA_RETRIES=5
export LLM_MODEL=llama3.1:latest
export EMBEDDINGS_PROVIDER=ollama
export EMBEDDINGS_MODEL=nomic-embed-text
```

### llama.cpp server

Start the server (OpenAI-compatible mode):

```bash
# Example (adjust to your setup)
./server -m /path/to/model.gguf --port 8080 --chatml
```

Env:

```bash
export LLM_PROVIDER=llamacpp
export LLM_MODEL=local-gguf
export LLAMACPP_BASE_URL=http://localhost:8080
```

## API Examples

### Ingest PDF

```bash
curl -F "file=@/path/to/file.pdf" http://localhost:8001/ingest/pdf
```

### Ingest FAQ CSV

CSV must have headers `question,answer`.

```bash
curl -F "file=@/path/to/faq.csv" http://localhost:8001/ingest/faq_csv
```

### Chat

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is this document about?","top_k":4}'
```

### Stats

```bash
curl http://localhost:8001/stats
```

## Testing Steps

1) Start the API on port 8001:

```bash
uvicorn app.main:app --reload --app-dir backend --port 8001
```

2) Open the UI in your browser:

```bash
open http://localhost:8010/
```

3) Ingest a PDF (API):

```bash
curl -F "file=@/path/to/file.pdf" http://localhost:8010/ingest/pdf
```

4) Ingest an FAQ CSV (API):

```bash
curl -F "file=@/path/to/faq.csv" http://localhost:8010/ingest/faq_csv
```

5) Verify stats:

```bash
curl http://localhost:8010/stats
```

6) Chat:

```bash
curl -X POST http://localhost:8010/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is this document about?","top_k":4}'
```

Optional unit tests:

```bash
pytest
```

## How it works

- **Chunking**: Uses `RecursiveCharacterTextSplitter` with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP` env vars.
- **Embeddings**: Uses Ollama embeddings (`nomic-embed-text`) or fallback to `sentence-transformers`.
- **Storage**: ChromaDB persists to `CHROMA_DIR` with per-chunk metadata (source, page, chunk_id, created_at, type).
- **Retrieval**: Similarity search with `top_k` results and scores from Chroma.
- **Prompting**: The prompt includes the question, retrieved context, and a citation list. The model must return JSON with `answer`, `citations`, `followups`.
- **Citations**: Response includes filename/page/chunk_id/score for each cited chunk. If context is insufficient, the answer is `I don't know based on the provided documents.` and citations are empty.

## Environment Variables

- `CHROMA_DIR` (default: `backend/.chroma`)
- `CHROMA_COLLECTION` (default: `documents`)
- `CHUNK_SIZE` (default: `800`)
- `CHUNK_OVERLAP` (default: `120`)
- `LLM_PROVIDER` (`ollama` | `llamacpp`)
- `LLM_MODEL` (model name; if `OLLAMA_MODEL` is set it takes precedence for Ollama)
- `OLLAMA_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `llama3.1:latest`)
- `OLLAMA_TIMEOUT` (default: `30`)
- `OLLAMA_RETRIES` (default: `5`)
- `LLAMACPP_BASE_URL` (default: `http://localhost:8080`)
- `EMBEDDINGS_PROVIDER` (`ollama` | `st`)
- `EMBEDDINGS_MODEL` (default: `nomic-embed-text`)
- `EMBEDDINGS_ST_MODEL` (default: `all-MiniLM-L6-v2`)

## Notes

- Tool calling is limited to `list_sources`, `get_context`, and `summarize_chunk`.
- Each request logs `request_id`, retrieved chunk IDs, model name, and latency.

## Concepts

See `README_CONCEPTS.md` for a breakdown of RAG, LangChain usage, embeddings, and retrieval.

## Troubleshooting

- If `/ingest/pdf` fails with `404` on `http://localhost:11434/api/embeddings`, either:
  - Ensure Ollama is running and `nomic-embed-text` is pulled, or
  - Set `EMBEDDINGS_PROVIDER=st` to use local sentence-transformers embeddings.

- If you see deprecation warnings for `HuggingFaceEmbeddings`, run:
  - `pip install -U langchain-huggingface`
