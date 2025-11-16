"""Protocols for Prompt Task Layer components."""

from __future__ import annotations

from typing import Protocol

from langformer.prompting.types import (
    PromptTaskResult,
    PromptTaskSpec,
    RenderedPrompt,
)


class PromptRenderer(Protocol):
    """Render a prompt for a given task specification."""

    def render(self, spec: PromptTaskSpec) -> RenderedPrompt:
        """Render a prompt for ``spec``."""
        ...


class PromptTaskEngine(Protocol):
    """Execute a task specification using an internal prompt engine."""

    def run(self, spec: PromptTaskSpec) -> PromptTaskResult:
        """Run ``spec`` and return a normalized result."""
        ...
