from __future__ import annotations

import json
from langchain_core.tools import tool

from app.rag import retrieve
from app.rag.llm import generate


@tool
def list_sources() -> str:
    """Return loaded sources and chunk counts."""
    stats = retrieve.get_stats()
    return json.dumps(stats, indent=2)


@tool
def get_context(query: str, top_k: int = 4) -> str:
    """Return retrieved chunks and metadata."""
    results = retrieve.retrieve(query, top_k=top_k)
    payload = []
    for doc, score in results:
        payload.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
        )
    return json.dumps(payload, indent=2)


@tool
def summarize_chunk(chunk_id: str) -> str:
    """Summarize a specific chunk by chunk_id."""
    result = retrieve.get_chunk_by_id(chunk_id)
    docs = result.get("documents", [])
    if not docs:
        return "Chunk not found."
    text = docs[0]
    prompt = (
        "Summarize the following chunk in 2-3 sentences.\n\n" + text
    )
    return generate(prompt)
