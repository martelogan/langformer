# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Logging helpers (redaction, rotating logs, stream dispatch)."""

from __future__ import annotations

import logging
import threading

from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_REDACT_KEYS = {
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}


def redact(text: str) -> str:
    """Scrub sensitive tokens from streamed text."""

    if not text:
        return ""
    cleaned = text
    for key in _REDACT_KEYS:
        if key in cleaned:
            cleaned = cleaned.replace(key, "<REDACTED_KEY>")
    return cleaned


def setup_file_logger(
    log_file: Path, name: str = "langformer"
) -> logging.Logger:
    """Configure a rotating file logger (idempotent per file)."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    marker = str(log_file)
    if not any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "_tap_tag", None) == marker
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=3
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        handler._tap_tag = marker  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


class StreamDispatcher:
    """Writes streamed LLM deltas to disk and/or stdout."""

    def __init__(
        self, log_path: Optional[Path], *, mode: str = "file"
    ) -> None:
        self.log_path = log_path
        self.mode = mode
        self._lock = threading.Lock()

    def emit(
        self, chunk: Optional[str], *, prefix: Optional[str] = None
    ) -> None:
        """Handle a streamed chunk."""

        if not chunk:
            return
        text = redact(chunk)
        if self.log_path is not None:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(text)
        if self.mode in {"stdout", "both"}:
            label = f"[{prefix}] " if prefix else ""
            try:
                print(f"{label}{text}", end="", flush=True)
            except Exception:
                pass
