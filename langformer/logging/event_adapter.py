# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Streaming adapter for OpenAI responses."""

from __future__ import annotations

import json
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class StreamDelta:
    ts: float
    kind: str
    data: dict[str, Any]


class EventAdapter:
    def __init__(
        self,
        model: str,
        store_responses: bool,
        timeout_s: int,
        jsonl_path: Path,
        stop_event: Optional[threading.Event] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.store_responses = store_responses
        self.timeout_s = timeout_s
        self.jsonl_path = jsonl_path
        self.stop_event = stop_event or threading.Event()
        self.on_delta = on_delta
        self._client = client
        self._buffer: list[str] = []
        self._buffer_bytes = 0
        self._last_flush = time.time()
        self._lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise RuntimeError(
                "OpenAI SDK not available. Install openai>=1.40."
            )
        self._client = OpenAI()
        return self._client

    def _record_event(
        self,
        kind: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        data = data or {}
        ev = StreamDelta(time.time(), kind, data)
        line = json.dumps(
            {"ts": ev.ts, "kind": ev.kind, "data": ev.data},
            ensure_ascii=False,
        )
        with self._lock:
            self._buffer.append(line)
            self._buffer_bytes += len(line) + 1

    def _should_flush(self) -> bool:
        now = time.time()
        if self._buffer_bytes >= 8 * 1024:
            return True
        if now - self._last_flush >= 0.05 and self._buffer:
            return True
        return False

    def _flush(self) -> None:
        if not self._buffer:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for line in self._buffer:
                handle.write(line + "\n")
        self._buffer.clear()
        self._buffer_bytes = 0
        self._last_flush = time.time()

    def _flusher(self, running_flag: threading.Event) -> None:
        while not running_flag.is_set():
            time.sleep(0.025)
            with self._lock:
                if self._should_flush():
                    self._flush()
        with self._lock:
            self._flush()

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        extras: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        client = self._ensure_client()
        output_parts: list[str] = []
        response_id: Optional[str] = None
        error_msg: Optional[str] = None

        done_flag = threading.Event()
        t = threading.Thread(
            target=self._flusher, args=(done_flag,), daemon=True
        )
        t.start()

        self._record_event("stream_started", {"model": self.model})

        params: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "timeout": self.timeout_s,
        }
        if extras:
            params.update(extras)
        if self.store_responses:
            params["store"] = True

        try:
            with client.responses.stream(
                **params,
            ) as stream:  # type: ignore[attr-defined]
                for event in stream:
                    if self.stop_event.is_set():
                        self._record_event("canceled")
                        break
                    kind = (
                        getattr(event, "type", None)
                        or getattr(event, "event", None)
                        or "unknown"
                    )
                    data: dict[str, Any] = {}
                    if kind == "response.output_text.delta":
                        delta = getattr(event, "delta", None)
                        if isinstance(delta, str) and delta:
                            output_parts.append(delta)
                            if self.on_delta:
                                try:
                                    self.on_delta(delta)
                                except Exception:
                                    pass
                            data["delta"] = delta
                    if kind == "response.completed" and hasattr(
                        event, "response"
                    ):
                        resp = getattr(event, "response")
                        response_id = getattr(resp, "id", None)
                        if response_id:
                            data["response_id"] = response_id
                    if kind == "response.error" and hasattr(event, "error"):
                        err = getattr(event, "error")
                        error_msg = getattr(err, "message", None)
                        data["error"] = error_msg
                    self._record_event(kind, data)
        except Exception as exc:  # pragma: no cover
            error_msg = str(exc)
            self._record_event("stream_error", {"error": error_msg})
        finally:
            done_flag.set()
            t.join()

        return {
            "output_text": "".join(output_parts),
            "response_id": response_id,
            "error": error_msg,
        }
