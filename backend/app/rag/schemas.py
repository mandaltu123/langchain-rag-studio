from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    chunks_added: int
    source: str


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    sources: list[dict[str, Any]]


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    top_k: Optional[int] = None
    source: Optional[str] = None


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
