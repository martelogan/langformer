from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from langformer.cli import _import_plugin_modules, main
from langformer.preprocessing.delegates import (
    DelegateRegistry,
    ExecutionDelegate,
)
from langformer.preprocessing.planners import (
    ExecutionPlan,
    ExecutionPlanner,
    PlannerRegistry,
)
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)


def test_cli_transpile_list_and_show(tmp_path: Path, monkeypatch):
    source = tmp_path / "input.py"
    source.write_text("def main():\n    return 1\n")
    output = tmp_path / "out.py"
    run_root = tmp_path / "runs"

    common_args = [
        str(source),
        "--target",
        str(output),
        "--run-root",
        str(run_root),
        "--config",
        "configs/default_config.yaml",
        "--llm-provider",
        "echo",
        "--model",
        "echo",
        "--parallel-workers",
        "2",
        "--max-retries",
        "2",
        "--stream-mode",
        "none",
        "--sandbox",
        "--deny-network",
        "--run-timeout",
        "5",
        "--api-base",
        "http://example.com",
        "--high-reasoning",
    ]
    exit_code = main(common_args)
    assert exit_code == 0
    assert output.exists()

    # list runs (should print but exit successfully)
    assert main(["--list-runs", "--run-root", str(run_root)]) == 0

    # show run details
    run_dir = next(run_root.iterdir())
    assert (
        main(
            [
                "--show-run",
                run_dir.name,
                "--run-root",
                str(run_root),
            ]
        )
        == 0
    )

    # resume existing run
    output.unlink()
    resume_args = common_args + ["--resume-run", run_dir.name]
    assert main(resume_args) == 0
    assert output.exists()


def test_cli_imports_plugin_modules(monkeypatch):
    registry = PlannerRegistry.get_registry()
    registry.unregister("sample_plugin_planner")
    monkeypatch.syspath_prepend(".")
    config = {
        "transpilation": {
            "plugins": {"modules": ["tests.sample_plugin"]},
        }
    }
    _import_plugin_modules(config, Path(".").resolve())
    planner_cls = registry.get("sample_plugin_planner")
    assert planner_cls.__name__ == "SamplePlanner"
    registry.unregister("sample_plugin_planner")


def test_cli_uses_layout_paths(tmp_path: Path):
    run_root = tmp_path / "runs"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    source = src_dir / "input.py"
    source.write_text("def main():\n    return 1\n")
    out_dir = tmp_path / "outputs"
    config_path = tmp_path / "layout.yaml"
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "python",
            "layout": {
                "input": {"path": str(source)},
                "output": {
                    "kind": "directory",
                    "path": str(out_dir),
                    "filename": "result.py",
                },
            },
            "runtime": {"enabled": True, "run_root": str(run_root)},
            "verification": {"strategy": "exact_match"},
            "llm": {"provider": "echo", "model": "echo"},
        }
    }
    config_path.write_text(yaml.safe_dump(config))

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--llm-provider",
            "echo",
            "--model",
            "echo",
        ]
    )
    assert exit_code == 0
    result_file = out_dir / "result.py"
    assert result_file.exists()


def test_cli_layout_relative_to_config(tmp_path: Path):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    source = config_dir / "input.py"
    source.write_text("def main():\n    return 1\n")
    run_root = tmp_path / "runs"
    config_path = config_dir / "config.yaml"
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "python",
            "layout": {
                "relative_to": "config",
                "input": {"path": "input.py"},
                "output": {
                    "kind": "directory",
                    "path": "outputs",
                    "filename": "result.py",
                },
            },
            "runtime": {"enabled": True, "run_root": str(run_root)},
            "verification": {"strategy": "exact_match"},
            "llm": {"provider": "echo", "model": "echo"},
        }
    }
    config_path.write_text(yaml.safe_dump(config))

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--llm-provider",
            "echo",
            "--model",
            "echo",
        ]
    )
    assert exit_code == 0
    result_file = config_dir / "outputs" / "result.py"
    assert result_file.exists()


def test_cli_delegate_executes_and_verifies(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.py"
    source.write_text("def main():\n    return 1\n")
    target = tmp_path / "output.py"
    run_root = tmp_path / "runs"
    run_root.mkdir()

    planner_name = "dummy_planner"
    delegate_name = "dummy_delegate"

    class DummyPlanner(ExecutionPlanner):
        def plan(
            self, source_path: Path, config: Dict[str, Any]
        ) -> ExecutionPlan:
            return ExecutionPlan(
                action="delegate", context={"delegate": delegate_name}
            )

    class DummyDelegate(ExecutionDelegate):
        executed = False
        verified = False

        def execute(
            self,
            source_path: Path,
            target_path: Path,
            run_root: Path,
            plan: ExecutionPlan,
            config: Dict[str, Any],
        ) -> int:
            DummyDelegate.executed = True
            target_path.write_text("print('ok')\n", encoding="utf-8")
            return 0

        def verify(
            self,
            unit: TranspileUnit,
            candidate: CandidatePatchSet,
            ctx: IntegrationContext,
        ) -> VerifyResult:
            DummyDelegate.verified = True
            return VerifyResult(passed=True, details={"unit": unit.id})

    planner_registry = PlannerRegistry.get_registry()
    delegate_registry = DelegateRegistry.get_registry()
    planner_registry.register(planner_name, DummyPlanner)
    delegate_registry.register(delegate_name, DummyDelegate)

    config_path = tmp_path / "delegate.yaml"
    config = {
        "transpilation": {
            "source_language": "python",
            "target_language": "python",
            "layout": {
                "input": {"path": str(source)},
                "output": {"kind": "file", "path": str(target)},
            },
            "planner": {"kind": planner_name},
            "runtime": {"enabled": True, "run_root": str(run_root)},
            "llm": {"provider": "echo", "model": "echo"},
        }
    }
    config_path.write_text(yaml.safe_dump(config))

    try:
        exit_code = main(["--config", str(config_path)])
        assert exit_code == 0
        assert DummyDelegate.executed is True
        assert DummyDelegate.verified is True
    finally:
        planner_registry.unregister(planner_name)
        delegate_registry.unregister(delegate_name)
