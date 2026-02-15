from __future__ import annotations

import json
import time
from typing import Any, Optional

import requests

from app.core.config import settings


class LLMError(RuntimeError):
    pass


def _ollama_chat(messages: list[dict[str, str]], temperature: float = 0.1) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    last_error: Optional[Exception] = None
    for attempt in range(settings.ollama_retries):
        try:
            response = requests.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=settings.ollama_timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < settings.ollama_retries - 1:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
    raise LLMError(f"Ollama unavailable: {last_error}")


def _llamacpp_chat(messages: list[dict[str, str]], temperature: float = 0.1) -> str:
    response = requests.post(
        f"{settings.llamacpp_base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def generate(prompt: str, system: Optional[str] = None, temperature: float = 0.1) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return _ollama_chat(messages, temperature=temperature)
    if provider == "llamacpp":
        return _llamacpp_chat(messages, temperature=temperature)
    raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}")


def generate_json(prompt: str, system: Optional[str] = None) -> Any:
    raw = generate(prompt, system=system, temperature=0.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Invalid JSON from model: {raw}") from exc
