"""Shared dataclasses for the Prompt Task Layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    """Single chat message exchanged with a model."""

    role: Role
    content: str
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptTaskSpec:
    """Minimal, open-ended description of a prompted task."""

    kind: str
    task_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RenderedPrompt:
    """Result of rendering a prompt via a PromptRenderer."""

    message: ChatMessage
    preview: Optional[str] = None
    template: Optional[str] = None


@dataclass(slots=True)
class PromptTaskResult:
    """Normalized result of executing a prompted task."""

    output_type: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
