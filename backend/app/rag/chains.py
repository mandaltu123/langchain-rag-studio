from __future__ import annotations

import json
import time
from typing import Any, Optional
from uuid import uuid4

from langchain_core.runnables import RunnableLambda

from app.core.config import settings
from app.rag.logging import log_run
from app.rag.llm import LLMError, generate, generate_json
from app.rag.retrieve import retrieve
from app.rag.schemas import ChatResponse, Citation
from app.rag.tools import get_context, list_sources, summarize_chunk

IDK_ANSWER = "I don't know based on the provided documents."


def _build_citations(results):
    citations = []
    for doc, score in results:
        meta = doc.metadata or {}
        citations.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id", ""),
                "score": float(score),
            }
        )
    return citations


def _context_block(results) -> str:
    parts = []
    for doc, score in results:
        meta = doc.metadata or {}
        parts.append(
            "\n".join(
                [
                    f"chunk_id: {meta.get('chunk_id', '')}",
                    f"source: {meta.get('source', 'unknown')}",
                    f"page: {meta.get('page', 'n/a')}",
                    f"score: {float(score):.4f}",
                    f"content: {doc.page_content}",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _followups_from_context(question: str, context: str) -> list[str]:
    prompt = (
        "Generate exactly 3 short follow-up questions that are grounded in the context. "
        "Return them as a JSON array of strings, no extra text.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    try:
        data = generate_json(prompt)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()][:3]
    except LLMError:
        pass
    raw = generate(
        "Provide 3 short follow-up questions, one per line.\n\nContext:\n"
        + context
    )
    return [line.strip("- ") for line in raw.splitlines() if line.strip()][:3]


def _rag_answer(message: str, results) -> ChatResponse:
    if not results:
        return ChatResponse(answer=IDK_ANSWER, citations=[], followups=[])

    citations = _build_citations(results)
    context = _context_block(results)

    prompt = (
        "You are a helpful assistant for question answering over provided documents. "
        "Use only the context. If the context is insufficient, answer exactly: "
        f"\"{IDK_ANSWER}\".\n\n"
        "Return a JSON object with keys: answer, citations, followups.\n"
        "Citations must be an array of objects from the provided citation list.\n"
        "Followups must be an array of 3 short questions grounded in context.\n\n"
        f"Question: {message}\n\n"
        f"Context:\n{context}\n\n"
        f"Citation list (select from these):\n{json.dumps(citations, indent=2)}"
    )

    def _attempt(prompt_text: str):
        return generate_json(prompt_text)

    data: Optional[Any] = None
    try:
        data = _attempt(prompt)
    except LLMError:
        data = None

    def _needs_retry(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return True
        answer = str(parsed.get("answer", "")).strip()
        citations_out = parsed.get("citations", [])
        if answer != IDK_ANSWER and not citations_out:
            return True
        return False

    if data is None or _needs_retry(data):
        fix_prompt = (
            "Fix the JSON output format. Ensure keys: answer, citations, followups. "
            "If you used the context, include citations from the list. "
            "If insufficient context, answer exactly: "
            f"\"{IDK_ANSWER}\" and return empty citations.\n\n"
            f"Question: {message}\n\n"
            f"Context:\n{context}\n\n"
            f"Citation list:\n{json.dumps(citations, indent=2)}"
        )
        try:
            data = _attempt(fix_prompt)
        except LLMError:
            data = {
                "answer": IDK_ANSWER,
                "citations": [],
                "followups": [],
            }

    answer = str(data.get("answer", "")).strip() if isinstance(data, dict) else ""
    citations_out = data.get("citations", []) if isinstance(data, dict) else []
    followups = data.get("followups", []) if isinstance(data, dict) else []

    if answer != IDK_ANSWER and not citations_out:
        citations_out = citations
    if not followups and results:
        followups = _followups_from_context(message, context)

    response = ChatResponse(
        answer=answer or IDK_ANSWER,
        citations=[Citation(**c) for c in citations_out] if citations_out else [],
        followups=[str(item) for item in followups][:3],
    )
    return response


def _route_action(message: str) -> dict[str, Any]:
    prompt = (
        "Decide which action to take based on the user message. "
        "Return ONLY valid JSON with keys: action, args. "
        "Actions: chat, list_sources, get_context, summarize_chunk. "
        "Use list_sources when asking for sources or stats. "
        "Use get_context when the user asks to see retrieved context. "
        "Use summarize_chunk when the user asks to summarize a specific chunk_id. "
        "Otherwise use chat.\n\n"
        f"Message: {message}"
    )
    try:
        data = generate_json(prompt)
        if isinstance(data, dict) and data.get("action"):
            return data
    except LLMError:
        pass

    lowered = message.lower()
    if "source" in lowered or "stats" in lowered:
        return {"action": "list_sources", "args": {}}
    if "context" in lowered:
        return {"action": "get_context", "args": {"query": message}}
    if "summar" in lowered and "chunk" in lowered:
        return {"action": "summarize_chunk", "args": {"chunk_id": ""}}
    return {"action": "chat", "args": {}}


def chat_with_tools(
    session_id: Optional[str],
    message: str,
    top_k: Optional[int],
    source: Optional[str] = None,
) -> ChatResponse:
    request_id = uuid4().hex
    start = time.perf_counter()
    router = RunnableLambda(_route_action)
    action_data = router.invoke(message)
    action = action_data.get("action", "chat")
    args = action_data.get("args", {}) or {}

    if action == "list_sources":
        answer = list_sources.invoke({})
        return ChatResponse(answer=answer, citations=[], followups=[])
    if action == "get_context":
        query = args.get("query", message)
        k = int(args.get("top_k", top_k or settings.top_k_default))
        answer = get_context.invoke({"query": query, "top_k": k})
        return ChatResponse(answer=answer, citations=[], followups=[])
    if action == "summarize_chunk":
        chunk_id = args.get("chunk_id", "")
        answer = summarize_chunk.invoke({"chunk_id": chunk_id})
        return ChatResponse(answer=answer, citations=[], followups=[])

    results = retrieve(message, top_k=top_k, source=source)
    response = _rag_answer(message, results)

    latency_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = [doc.metadata.get("chunk_id", "") for doc, _ in results]
    log_run(request_id, settings.llm_model, latency_ms, retrieved_ids)
    return response
