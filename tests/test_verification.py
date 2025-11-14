from pathlib import Path

from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.runtime.runner.manager import RunnerManager
from langformer.runtime.runner.sandbox import SandboxRunner
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    Oracle,
    TranspileUnit,
    VerifyResult,
)
from langformer.verification.strategies import (
    CustomOracleStrategy,
    ExecutionMatchStrategy,
)


def _unit(source: str) -> TranspileUnit:
    return TranspileUnit(id="u1", language="python", source_code=source)


def test_execution_strategy_passes_for_identical_python(
    tmp_path: Path,
) -> None:
    plugin = LightweightPythonLanguagePlugin()
    code = """\
def main(a: int = 1, b: int = 2):
    return a + b
"""
    unit = _unit(code)
    candidate = CandidatePatchSet(files={tmp_path / "out.py": code})
    ctx = IntegrationContext(
        target_language="python",
        layout={"output": {"path": str(tmp_path / "out.py"), "kind": "file"}},
    )
    strategy = ExecutionMatchStrategy(test_inputs=[{"a": 3, "b": 5}, {}])

    result = strategy.verify(
        unit, candidate, ctx, source_plugin=plugin, target_plugin=plugin
    )
    assert result.passed


def test_execution_strategy_detects_difference(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    source = """\
def main() -> int:
    return 2
"""
    target = """\
def main() -> int:
    return 5
"""
    unit = _unit(source)
    candidate = CandidatePatchSet(files={tmp_path / "out.py": target})
    ctx = IntegrationContext(
        target_language="python",
        layout={"output": {"path": str(tmp_path / "out.py"), "kind": "file"}},
    )
    strategy = ExecutionMatchStrategy(test_inputs=[{}])

    result = strategy.verify(
        unit, candidate, ctx, source_plugin=plugin, target_plugin=plugin
    )
    assert not result.passed


def test_execution_strategy_with_sandbox_runner(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    code = """\
print("ALL_TESTS_PASSED")
"""
    unit = _unit(code)
    candidate = CandidatePatchSet(files={tmp_path / "out.py": code})
    runner = SandboxRunner(tmp_path / "runs", timeout_s=5)
    manager = RunnerManager(runner=runner)
    ctx = IntegrationContext(
        target_language="python",
        layout={"output": {"path": str(tmp_path / "out.py"), "kind": "file"}},
    )
    strategy = ExecutionMatchStrategy(test_inputs=[{}], runner_manager=manager)

    result = strategy.verify(
        unit, candidate, ctx, source_plugin=plugin, target_plugin=plugin
    )
    assert result.passed


def test_custom_oracle_strategy_uses_context(tmp_path: Path) -> None:
    plugin = LightweightPythonLanguagePlugin()
    unit = _unit("def main():\n    return 1\n")
    candidate = CandidatePatchSet(
        files={tmp_path / "out.py": "print('hello')"}
    )

    def contains_print(source_code: str, target_code: str, metadata):
        passed = "print" in target_code
        return VerifyResult(passed=passed, details={"unit": metadata["unit"]})

    ctx = IntegrationContext(
        target_language="python",
        layout={"output": {"path": str(tmp_path / "out.py"), "kind": "file"}},
        oracle=Oracle(verify=contains_print),
    )
    strategy = CustomOracleStrategy()

    result = strategy.verify(
        unit, candidate, ctx, source_plugin=plugin, target_plugin=plugin
    )
    assert result.passed
