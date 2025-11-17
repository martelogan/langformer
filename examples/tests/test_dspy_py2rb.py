from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "dspy",
    reason="DSPy is required for the DSPy Py→Rb example tests.",
)

from examples.dspy_py2rb_transpiler.agent import Py2RbDSPyTranspilerAgent
from examples.dspy_py2rb_transpiler.dspy_components import (
    DSPyFeedbackOptimizer,
)
from langformer.agents.base import LLMConfig
from langformer.artifacts import ArtifactManager
from langformer.configuration import ArtifactSettings
from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.prompting.manager import PromptManager
from langformer.types import (
    IntegrationContext,
    LayoutPlan,
    TranspileUnit,
    VerifyResult,
)

PROMPT_DIR = Path("langformer/prompting/templates")


def _llm_config(tmp_path: Path, provider) -> LLMConfig:
    manager = PromptManager(PROMPT_DIR)
    artifact_manager = ArtifactManager(
        ArtifactSettings(root=(tmp_path / "artifacts").resolve())
    )
    return LLMConfig(
        provider=provider,
        prompt_manager=manager,
        prompt_paths=manager.search_paths,
        artifact_manager=artifact_manager,
    )


class _Provider:
    def generate(self, prompt: str, **kwargs):
        return prompt


class _Verifier:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls = 0

    def verify(self, unit, candidate, ctx):
        self.calls += 1
        text = next(iter(candidate.files.values()))
        passed = self.keyword in text
        details = {"failures": []}
        if not passed:
            details["failures"].append("missing multiplier x2")
        return VerifyResult(passed=passed, details=details)


def _module_factory():
    class _Module:
        def __call__(self, **kwargs):
            metadata = kwargs.get("unit_metadata") or {}
            multiplier = metadata.get("optimizer_multiplier", 1)
            feedback = kwargs.get("verification_feedback") or []
            if feedback:
                failures = feedback[-1].get("failures") or []
                if any("missing multiplier" in item for item in failures):
                    multiplier = 2
            text = f"def scale(score)\n  score * {multiplier}\nend"

            class _Result:
                output = text

            return _Result()

    return _Module


def _unit() -> TranspileUnit:
    return TranspileUnit(
        id="unit",
        language="python",
        source_code="def scale_score(x):\n    return x * 2\n",
    )


def _ctx(tmp_path: Path) -> IntegrationContext:
    return IntegrationContext(
        target_language="ruby",
        layout=LayoutPlan(
            {"output": {"path": str(tmp_path / "out.rb"), "kind": "file"}}
        ),
        feature_spec={
            "ruby_namespace": "Report",
            "verification_cases": [{"values": [1, 2, 3]}],
            "ruby_tests": ["tests/report_module_test.rb"],
        },
    )


def test_dspy_feedback_optimizer_records_artifacts(tmp_path: Path) -> None:
    manager = ArtifactManager(
        ArtifactSettings(root=(tmp_path / "artifacts").resolve())
    )
    optimizer = DSPyFeedbackOptimizer(artifact_manager=manager)

    optimizer.record_feedback(
        "unit",
        1,
        {
            "failures": ["Ruby outputs did not match Python results."],
            "cases": [{"values": [1, 2]}],
            "python": [{"scaled": [2, 4], "report": "2, 4"}],
            "ruby": [{"scaled": [1, 3], "report": "1, 3"}],
            "ruby_tests": [
                {"name": "report_test.rb", "status": "failed", "message": "boom"}
            ],
            "syntax": {"passed": False, "error": "unexpected end-of-input"},
        },
    )
    overrides = optimizer.build_overrides("unit")

    assert overrides["hints"]
    assert overrides["style_guide"]
    assert "cases" in overrides["verifier_requirements"]
    manifest = manager.manifest_for("unit")
    assert "transpiler" in manifest


def test_py2rb_dspy_agent_uses_optimizer(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    cfg = _llm_config(tmp_path, _Provider())
    agent = Py2RbDSPyTranspilerAgent(
        plugin,
        plugin,
        llm_config=cfg,
        max_retries=2,
        module_factory=_module_factory(),
    )
    verifier = _Verifier(keyword="* 2")

    candidate = agent.transpile(_unit(), _ctx(tmp_path), verifier=verifier)

    assert "* 2" in next(iter(candidate.files.values()))
    assert verifier.calls == 2
    manifest = cfg.artifact_manager.manifest_for("unit")
    assert "transpiler" in manifest
