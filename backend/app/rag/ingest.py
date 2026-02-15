from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import BytesIO, StringIO
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import settings
from app.rag.embeddings import get_embeddings
from app.rag.store import get_vector_store


def _text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    reader = PdfReader(BytesIO(file_bytes))
    docs: list[Document] = []
    splitter = _text_splitter()
    created_at = datetime.now(timezone.utc).isoformat()

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        page_docs = splitter.split_text(text)
        for chunk_index, chunk in enumerate(page_docs):
            chunk_id = f"{uuid4().hex}"
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_index,
                        "chunk_id": chunk_id,
                        "created_at": created_at,
                        "type": "pdf",
                    },
                )
            )

    if not docs:
        return 0

    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    store.add_documents(docs)
    return len(docs)


def ingest_faq_csv(file_bytes: bytes, filename: str) -> int:
    content = file_bytes.decode("utf-8")
    reader = csv.DictReader(StringIO(content))
    docs: list[Document] = []
    splitter = _text_splitter()
    created_at = datetime.now(timezone.utc).isoformat()

    for row in reader:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question and not answer:
            continue
        text = f"Q: {question}\nA: {answer}".strip()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            chunk_id = f"{uuid4().hex}"
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk_id": chunk_id,
                        "created_at": created_at,
                        "type": "faq",
                        "question": question,
                    },
                )
            )

    if not docs:
        return 0

    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    store.add_documents(docs)
    return len(docs)
