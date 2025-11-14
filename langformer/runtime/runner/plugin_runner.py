"""Runner that delegates execution to language plugins."""

from __future__ import annotations

from typing import Any, Dict

from langformer.languages.base import LanguagePlugin

from .base import ExecutionRunner, RunResult


class PluginRunner(ExecutionRunner):
    """Executes code by calling LanguagePlugin.compile/execute with error handling."""

    def run(
        self,
        plugin: LanguagePlugin,
        code: str,
        inputs: Dict[str, Any] | None = None,
    ) -> RunResult:
        try:
            if hasattr(plugin, "compile"):
                plugin.compile(code)
            payload = plugin.execute(code, inputs or {})
            return RunResult(success=True, output=payload)
        except Exception as exc:  # pragma: no cover - defensive path
            return RunResult(success=False, output=None, error=str(exc))
