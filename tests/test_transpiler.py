from __future__ import annotations

import threading

from pathlib import Path
from typing import List

import pytest

from langformer.agents.base import LLMConfig
from langformer.agents.transpiler import LLMTranspilerAgent
from langformer.exceptions import TranspilationAttemptError
from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.llm.providers import LLMProvider
from langformer.prompting.manager import PromptManager
from langformer.types import (
    IntegrationContext,
    LayoutPlan,
    TranspileUnit,
    VerifyResult,
)

PROMPT_DIR = Path("langformer/prompting/templates")


def _llm_config_for(provider: LLMProvider) -> LLMConfig:
    manager = PromptManager(PROMPT_DIR)
    return LLMConfig(
        provider=provider,
        prompt_manager=manager,
        prompt_paths=manager.search_paths,
    )


def _make_agent(
    provider: LLMProvider, plugin: LightweightPythonLanguagePlugin, **kwargs
) -> LLMTranspilerAgent:
    return LLMTranspilerAgent(
        plugin,
        plugin,
        llm_config=_llm_config_for(provider),
        **kwargs,
    )


class _StubLLM:
    def __init__(
        self,
        responses: List[str],
        *,
        parallel: bool = False,
        success_threshold: float = 1.0,
    ) -> None:
        self._responses = responses
        self._idx = 0
        self._lock = threading.Lock()
        self._parallel = parallel
        self._success_threshold = success_threshold

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature")
        if (
            self._parallel
            and temperature is not None
            and temperature >= self._success_threshold
        ):
            return "def ok():\n    return 42\n"
        with self._lock:
            if self._idx >= len(self._responses):
                return self._responses[-1]
            value = self._responses[self._idx]
            self._idx += 1
        return value


class _Verifier:
    def __init__(
        self, *, success_keyword: str, source_plugin, target_plugin
    ) -> None:
        self.success_keyword = success_keyword
        self.calls = 0
        self.source_plugin = source_plugin
        self.target_plugin = target_plugin

    def verify(self, unit, candidate, ctx):
        self.calls += 1
        contents = next(iter(candidate.files.values()))
        passed = self.success_keyword in contents
        return VerifyResult(
            passed=passed, details={"passed": passed, "attempts": self.calls}
        )


def _unit() -> TranspileUnit:
    return TranspileUnit(
        id="u", language="python", source_code="def main():\n    return 1\n"
    )


def _ctx(tmp_path: Path) -> IntegrationContext:
    return IntegrationContext(
        target_language="python",
        layout=LayoutPlan(
            {"output": {"path": str(tmp_path / "out.py"), "kind": "file"}}
        ),
    )


def test_transpiler_retries_until_success(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    llm = _StubLLM(
        ["def bad():\n    return 0\n", "def good():\n    return 1\n"]
    )
    agent = _make_agent(
        llm,
        plugin,
        max_retries=2,
        parallel_workers=1,
        temperature_range=(0.1, 0.9),
    )
    verifier = _Verifier(
        success_keyword="good", source_plugin=plugin, target_plugin=plugin
    )

    candidate = agent.transpile(_unit(), _ctx(tmp_path), verifier=verifier)

    assert "good" in next(iter(candidate.files.values()))
    assert verifier.calls == 2


def test_transpiler_parallel_path_returns_first_success(
    tmp_path: Path,
) -> None:
    plugin = LightweightPythonLanguagePlugin()
    llm = _StubLLM([], parallel=True, success_threshold=0.8)
    agent = _make_agent(
        llm,
        plugin,
        max_retries=2,
        parallel_workers=2,
        temperature_range=(0.2, 1.0),
    )
    verifier = _Verifier(
        success_keyword="ok", source_plugin=plugin, target_plugin=plugin
    )

    candidate = agent.transpile(_unit(), _ctx(tmp_path), verifier=verifier)

    assert "ok" in next(iter(candidate.files.values()))
    assert verifier.calls == 1


class _DummyProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        metadata = kwargs.get("metadata") or {}
        return (metadata.get("source_code") or prompt) + "\n# ok"


def test_transpiler_uses_prompt_manager(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    provider = _DummyProvider()
    agent = _make_agent(provider, plugin, max_retries=1)
    verifier = _Verifier(
        success_keyword="# ok", source_plugin=plugin, target_plugin=plugin
    )

    candidate = agent.transpile(_unit(), _ctx(tmp_path), verifier=verifier)

    assert "# ok" in next(iter(candidate.files.values()))
    assert provider.calls == 1


def test_transpiler_prefers_event_stream_output(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    provider = _DummyProvider()

    def _factory(unit, ctx, attempt, variant_label):
        class _Adapter:
            def stream(self, **kwargs):
                return {
                    "output_text": "def streamed():\n    return 5\n",
                    "response_id": "stream-1",
                }

        return _Adapter()

    agent = _make_agent(
        provider, plugin, max_retries=1, event_adapter_factory=_factory
    )
    verifier = _Verifier(
        success_keyword="streamed", source_plugin=plugin, target_plugin=plugin
    )

    candidate = agent.transpile(
        _unit(),
        _ctx(tmp_path),
        verifier=verifier,
        variant_label="orchestrator",
    )

    contents = next(iter(candidate.files.values()))
    assert "streamed" in contents
    assert provider.calls == 0
    assert candidate.notes.get("stream", {}).get("response_id") == "stream-1"


def test_transpiler_falls_back_when_stream_fails(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    provider = _DummyProvider()

    def _factory(unit, ctx, attempt, variant_label):
        class _Adapter:
            def stream(self, **kwargs):
                raise RuntimeError("stream boom")

        return _Adapter()

    agent = _make_agent(
        provider, plugin, max_retries=1, event_adapter_factory=_factory
    )
    verifier = _Verifier(
        success_keyword="# ok", source_plugin=plugin, target_plugin=plugin
    )

    candidate = agent.transpile(_unit(), _ctx(tmp_path), verifier=verifier)

    assert "# ok" in next(iter(candidate.files.values()))
    assert provider.calls == 1
    assert "stream" in candidate.notes
    assert candidate.notes["stream"]["error"]


def test_transpiler_skips_duplicate_same_worker(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    llm = _StubLLM(
        ["def dup():\n    return 0\n", "def unique():\n    return 2\n"]
    )
    agent = _make_agent(llm, plugin, max_retries=2)

    def _dedup_handler(code: str, attempt: int, variant_label: str | None):
        status = "duplicate_same_worker" if attempt == 1 else "unique"
        return {"status": status}

    verifier = _Verifier(
        success_keyword="unique", source_plugin=plugin, target_plugin=plugin
    )

    candidate = agent.transpile(
        _unit(),
        _ctx(tmp_path),
        verifier=verifier,
        dedup_handler=_dedup_handler,
    )

    assert candidate.notes["dedup"]["status"] == "unique"
    assert "unique" in next(iter(candidate.files.values()))


def test_transpiler_raises_on_cross_worker_dup(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    llm = _StubLLM(["def dup():\n    return 0\n"])
    agent = _make_agent(llm, plugin, max_retries=1)

    def _dedup_handler(code: str, attempt: int, variant_label: str | None):
        return {"status": "duplicate_cross_worker"}

    verifier = _Verifier(
        success_keyword="dup", source_plugin=plugin, target_plugin=plugin
    )

    with pytest.raises(TranspilationAttemptError):
        agent.transpile(
            _unit(),
            _ctx(tmp_path),
            verifier=verifier,
            dedup_handler=_dedup_handler,
        )


def test_transpiler_checks_cancel_event(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    llm = _StubLLM(["def nope():\n    return 0\n"])
    agent = _make_agent(llm, plugin)
    verifier = _Verifier(
        success_keyword="nope", source_plugin=plugin, target_plugin=plugin
    )

    event = threading.Event()
    event.set()

    with pytest.raises(TranspilationAttemptError):
        agent.transpile(
            _unit(), _ctx(tmp_path), verifier=verifier, cancel_event=event
        )
