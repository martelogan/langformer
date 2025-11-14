"""Execution runner interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langformer.languages.base import LanguagePlugin


@dataclass
class RunResult:
    success: bool
    output: Any = None
    error: str | None = None


class ExecutionRunner(Protocol):
    def run(
        self,
        plugin: LanguagePlugin,
        code: str,
        inputs: dict[str, Any] | None = None,
    ) -> RunResult:
        """Execute code using the provided plugin and return the result."""
        ...
