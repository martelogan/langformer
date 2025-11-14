"""Simple runner manager that mirrors KernelAgent's worker pattern."""

from __future__ import annotations

import queue
import threading

from typing import Any, Dict

from langformer.languages.base import LanguagePlugin

from .base import ExecutionRunner, RunResult
from .plugin_runner import PluginRunner


class RunnerManager:
    """Executes runner calls with optional timeouts in background threads."""

    def __init__(
        self,
        runner: ExecutionRunner | None = None,
        *,
        timeout: float | None = None,
    ) -> None:
        self._runner = runner or PluginRunner()
        self._timeout = timeout

    def run(
        self,
        plugin: LanguagePlugin,
        code: str,
        inputs: Dict[str, Any] | None = None,
    ) -> RunResult:
        result_queue: queue.Queue[RunResult] = queue.Queue(maxsize=1)

        def _target() -> None:
            result_queue.put(self._runner.run(plugin, code, inputs or {}))

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(self._timeout)
        if worker.is_alive():
            return RunResult(
                success=False, output=None, error="runner timeout"
            )
        return result_queue.get()
