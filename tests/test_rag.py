from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.rag import ingest, retrieve


class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def test_ingest_and_retrieve(tmp_path, monkeypatch):
    settings.chroma_dir = str(tmp_path / "chroma")
    monkeypatch.setattr(ingest, "get_embeddings", lambda: DummyEmbeddings())
    monkeypatch.setattr(retrieve, "get_embeddings", lambda: DummyEmbeddings())

    csv_bytes = b"question,answer\nWhat is RAG?,Retrieval augmented generation.\n"
    chunks = ingest.ingest_faq_csv(csv_bytes, "faq.csv")
    assert chunks > 0

    results = retrieve.retrieve("What is RAG?", top_k=1)
    assert results
    doc, score = results[0]
    assert "Retrieval augmented" in doc.page_content
    assert isinstance(score, float)
