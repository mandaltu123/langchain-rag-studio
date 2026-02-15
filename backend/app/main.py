from __future__ import annotations

import os
from typing import Optional

from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.rag.chains import chat_with_tools
from app.rag.ingest import ingest_faq_csv, ingest_pdf
from app.rag.retrieve import get_stats, get_suggestions
from app.rag.schemas import ChatRequest, ChatResponse, IngestResponse, StatsResponse

app = FastAPI(title=settings.app_name, version="0.1.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    content = await file.read()
    chunks = ingest_pdf(content, file.filename)
    return IngestResponse(chunks_added=chunks, source=file.filename)


@app.post("/ingest/faq_csv", response_model=IngestResponse)
async def ingest_faq_csv_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    content = await file.read()
    chunks = ingest_faq_csv(content, file.filename)
    return IngestResponse(chunks_added=chunks, source=file.filename)


@app.get("/stats", response_model=StatsResponse)
async def stats_endpoint() -> StatsResponse:
    stats = get_stats()
    return StatsResponse(**stats)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    return chat_with_tools(
        payload.session_id, payload.message, payload.top_k, payload.source
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/suggestions")
async def suggestions(source: Optional[str] = None) -> dict:
    return {"suggestions": get_suggestions(source=source)}


def _index_path() -> Optional[str]:
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return index_path
    return None


@app.get("/", response_class=HTMLResponse)
async def root() -> Response:
    index_path = _index_path()
    if index_path:
        return FileResponse(index_path, media_type="text/html")
    return RedirectResponse(url="/docs")


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> Response:
    index_path = _index_path()
    if index_path:
        return FileResponse(index_path, media_type="text/html")
    return RedirectResponse(url="/docs")

try:
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
except RuntimeError:
    pass
