from __future__ import annotations

import requests
from requests import Response
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.rag.logging import get_logger


class OllamaEmbeddings(Embeddings):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.logger = get_logger("embeddings")

    def _post(self, endpoint: str, payload: dict) -> Response:
        response = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response

    def _embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        try:
            data = self._post("/api/embeddings", payload).json()
            return data["embedding"]
        except requests.exceptions.RequestException:
            # Older Ollama versions use /api/embed
            data = self._post("/api/embed", payload).json()
            return data["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FallbackEmbeddings(Embeddings):
    def __init__(self, primary: Embeddings, fallback: Embeddings) -> None:
        self.primary = primary
        self.fallback = fallback
        self.logger = get_logger("embeddings")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.primary.embed_documents(texts)
        except requests.exceptions.RequestException as exc:
            self.logger.warning(
                "Primary embeddings failed; using fallback. error=%s", exc
            )
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.primary.embed_query(text)
        except requests.exceptions.RequestException as exc:
            self.logger.warning(
                "Primary embeddings failed; using fallback. error=%s", exc
            )
            return self.fallback.embed_query(text)


def get_embeddings() -> Embeddings:
    provider = settings.embeddings_provider.lower()
    if provider == "ollama":
        primary = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embeddings_model,
        )
        fallback = HuggingFaceEmbeddings(model_name=settings.embeddings_st_model)
        return FallbackEmbeddings(primary, fallback)
    return HuggingFaceEmbeddings(model_name=settings.embeddings_st_model)
