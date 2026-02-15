from __future__ import annotations

from typing import Iterable, Optional

from app.core.config import settings
from app.rag.embeddings import get_embeddings
from app.rag.store import get_vector_store


def retrieve(query: str, top_k: Optional[int] = None, source: Optional[str] = None):
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    k = top_k or settings.top_k_default
    if source:
        return store.similarity_search_with_score(query, k=k, filter={"source": source})
    return store.similarity_search_with_score(query, k=k)


def get_stats() -> dict:
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    collection = store._collection

    total_chunks = collection.count()
    metadata = collection.get(include=["metadatas"])
    metadatas = metadata.get("metadatas", [])

    sources: dict[str, dict] = {}
    for meta in metadatas:
        source = meta.get("source", "unknown")
        sources.setdefault(source, {"source": source, "chunks": 0})
        sources[source]["chunks"] += 1

    return {
        "documents": len(sources),
        "chunks": total_chunks,
        "sources": list(sources.values()),
    }


def get_chunks_by_ids(chunk_ids: Iterable[str]):
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    collection = store._collection
    result = collection.get(where={"chunk_id": {"$in": list(chunk_ids)}})
    return result


def get_chunk_by_id(chunk_id: str):
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    collection = store._collection
    result = collection.get(where={"chunk_id": chunk_id})
    return result


def get_suggestions(source: Optional[str] = None, limit: int = 5) -> list[str]:
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    collection = store._collection

    where_clause = {"source": source} if source else None
    data = collection.get(
        where=where_clause,
        include=["documents", "metadatas"],
    )
    docs = data.get("documents", []) or []
    if not docs:
        return []

    word_counts: dict[str, int] = {}
    for text in docs:
        for token in text.split():
            word = token.strip(".,;:!?()[]{}\"'").lower()
            if len(word) < 4 or not word.isalpha():
                continue
            word_counts[word] = word_counts.get(word, 0) + 1

    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in top_words[: max(limit, 5)]]
    if not keywords:
        return []

    templates = [
        "What is {k}?",
        "How does {k} work?",
        "Summarize {k}.",
        "Explain {k} in simple terms.",
        "What are the key points about {k}?",
    ]
    suggestions: list[str] = []
    for idx, word in enumerate(keywords[:limit]):
        suggestions.append(templates[idx % len(templates)].format(k=word))
    return suggestions
