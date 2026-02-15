from __future__ import annotations

import os
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.config import settings


def get_vector_store(embeddings: Embeddings) -> Chroma:
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )
