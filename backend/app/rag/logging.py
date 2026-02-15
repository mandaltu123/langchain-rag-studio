from __future__ import annotations

import logging
from typing import Iterable

from app.core.config import settings


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    return logger


def log_run(
    request_id: str,
    model: str,
    latency_ms: float,
    retrieved_chunk_ids: Iterable[str],
) -> None:
    logger = get_logger()
    chunk_list = list(retrieved_chunk_ids)
    logger.info(
        "request_id=%s model=%s latency_ms=%.2f retrieved_chunks=%s",
        request_id,
        model,
        latency_ms,
        chunk_list,
    )
