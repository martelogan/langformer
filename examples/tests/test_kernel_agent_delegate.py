from pathlib import Path

from examples.kernel_agent_delegate import (
    KernelAgentAutoPlanner,
    delegate as delegate_mod,
)
from examples.kernel_agent_delegate.delegate import KernelAgentDelegate
from langformer.preprocessing.delegates import ExecutionPlan
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)
from langformer.verification.oracles import OracleRegistry


def _base_config(run_root: Path, mode: str) -> dict:
    return {
        "examples": {"kernel_agent_delegate": {"mode": mode}},
        "transpilation": {"runtime": {"run_root": str(run_root)}},
    }


def test_kernel_agent_delegate_stub_mode_records_run(tmp_path: Path):
    source = tmp_path / "input.py"
    source.write_text("def main():\n    return 1\n")
    target = tmp_path / "out.py"
    run_root = tmp_path / "runs"
    config = _base_config(run_root, "stub")
    delegate = KernelAgentDelegate(config)
    plan = ExecutionPlan(
        action="delegate",
        context={"delegate": "kernel_agent_delegate", "solver": "kernelagent"},
    )

    exit_code = delegate.execute(source, target, run_root, plan, config)

    assert exit_code == 0
    assert target.read_text().startswith("# delegated stub")
    run_dirs = list(run_root.iterdir())
    assert run_dirs, "run session directory expected"


def test_kernel_agent_delegate_pipeline_invokes_runner(monkeypatch, tmp_path: Path):
    source = tmp_path / "problem.py"
    source.write_text("def main(x):\n    return x + 1\n")
    target = tmp_path / "compiled.py"
    run_root = tmp_path / "runs"
    composed = tmp_path / "composed.py"
    composed.write_text("print('kernel')\n")

    captured: dict = {}

    def _fake_run_pipeline(**kwargs):
        captured["kwargs"] = kwargs
        return {
            "run_dir": str(tmp_path / "fuser_run"),
            "subgraphs": "subgraphs.json",
            "kernels_summary": "summary.json",
            "composition": {"composed_path": str(composed)},
        }

    monkeypatch.setattr(delegate_mod, "_kernel_run_pipeline", _fake_run_pipeline)

    config = {
        "examples": {
            "kernel_agent_delegate": {
                "mode": "pipeline",
                "pipeline": {"extract_model": "model-x", "workers": 2, "verify": False},
            }
        },
        "transpilation": {"runtime": {"run_root": str(run_root)}},
    }
    delegate = KernelAgentDelegate(config)
    plan = ExecutionPlan(
        action="delegate",
        context={"delegate": "kernel_agent_delegate", "solver": "fuser"},
    )

    exit_code = delegate.execute(source, target, run_root, plan, config)

    assert exit_code == 0
    assert target.read_text() == composed.read_text()
    kwargs = captured["kwargs"]
    assert kwargs["problem_path"] == source.resolve()
    assert kwargs["workers"] == 2
    assert kwargs["dispatch_jobs"] == "auto"
    out_root = kwargs["out_root"]
    assert out_root.parent.parent == run_root


def test_kernel_agent_planner_delegate_registered(tmp_path: Path):
    source = tmp_path / "conv.py"
    source.write_text(
        "import torch.nn.functional as F\n"
        "def main(x):\n"
        "    return F.conv_transpose2d(x, x)\n"
    )
    planner = KernelAgentAutoPlanner({"enforce": True, "cache_path": str(tmp_path / "cache.json")})
    plan = planner.plan(source, {})
    assert plan.action == "delegate"


def test_delegate_router_config_overrides(tmp_path: Path):
    delegate = KernelAgentDelegate({"examples": {"kernel_agent_delegate": {}}})
    context = {
        "router_config": {
            "ka_max_rounds": 5,
            "ka_num_workers": 3,
            "ka_model": "router-ka",
            "fuser_dispatch_jobs": 7,
            "compose_max_iters": 11,
            "fuser_verify": False,
            "llm_models": {
                "extract": "extract-x",
                "dispatch": "dispatch-y",
                "compose": "compose-z",
                "ka_model": "router-llm",
            },
        }
    }

    kernel_cfg = delegate._build_kernel_config(context)
    pipeline_cfg = delegate._build_pipeline_config(context)

    assert kernel_cfg["max_rounds"] == 5
    assert kernel_cfg["workers"] == 3
    assert kernel_cfg["model"] == "router-llm"
    assert pipeline_cfg["dispatch_jobs"] == 7
    assert pipeline_cfg["compose_max_iters"] == 11
    assert pipeline_cfg["verify"] is False
    assert pipeline_cfg["extract_model"] == "extract-x"
    assert pipeline_cfg["dispatch_model"] == "dispatch-y"
    assert pipeline_cfg["compose_model"] == "compose-z"


def test_kernel_agent_oracle_invokes_runner(monkeypatch, tmp_path: Path):
    registry = OracleRegistry.get_registry()
    factory = registry.get("kernel_agent_runner")
    assert factory is not None

    stdout = tmp_path / "stdout.txt"
    stderr = tmp_path / "stderr.txt"
    stdout.write_text("ALL_TESTS_PASSED\n")
    stderr.write_text("")

    class DummyResult:
        def __init__(self) -> None:
            self.rc = 0
            self.passed = True
            self.validator_used = "sentinel"
            self.reason = "ALL_TESTS_PASSED"
            self.stdout_path = stdout
            self.stderr_path = stderr

    captured = {}

    def _fake_run_candidate(
        artifacts_code_path: Path,
        run_root: Path,
        timeout_s: int,
        isolated: bool,
        deny_network: bool,
        cancel_event=None,
    ):
        captured.update(
            {
                "path": artifacts_code_path,
                "run_root": run_root,
                "timeout_s": timeout_s,
                "isolated": isolated,
                "deny_network": deny_network,
            }
        )
        return DummyResult()

    monkeypatch.setattr(
        "examples.kernel_agent_delegate.oracle.run_candidate",
        _fake_run_candidate,
    )

    oracle = factory(
        {
            "run_root": str(tmp_path / "ka_runs"),
            "timeout_s": 42,
            "isolated": False,
            "deny_network": False,
        }
    )
    result = oracle.verify("src", "print('ok')", {"unit": "unit-1"})

    assert result.passed is True
    assert captured["run_root"] == (tmp_path / "ka_runs")
    assert captured["timeout_s"] == 42
    assert captured["isolated"] is False
    assert captured["deny_network"] is False


def test_kernel_agent_delegate_verify_uses_oracle(tmp_path: Path):
    delegate = KernelAgentDelegate({})

    class DummyOracle:
        def __init__(self) -> None:
            self.called = False

        def verify(self, source_code: str, target_code: str, metadata):
            self.called = True
            assert "delegate" in metadata
            return VerifyResult(passed=True, details={"source": source_code, "target": target_code})

    oracle = DummyOracle()
    ctx = IntegrationContext(target_language="python", oracle=oracle)
    unit = TranspileUnit(id="unit", language="python", source_code="print('source')")
    candidate = CandidatePatchSet(files={tmp_path / "out.py": "print('target')"})

    result = delegate.verify(unit, candidate, ctx)
    assert result.passed is True
    assert oracle.called is True


def test_kernel_agent_oracle_kernel_worker_mode(monkeypatch, tmp_path: Path):
    registry = OracleRegistry.get_registry()
    factory = registry.get("kernel_agent_runner")
    assert factory is not None

    class DummyWorker:
        def __init__(self, **kwargs):
            DummyWorker.kwargs = kwargs

        def run(self, kernel_code, test_code, problem_description, success_event):  # noqa: D401
            DummyWorker.ran = True
            assert "target" in kernel_code
            assert "ASSERT" in test_code
            assert "problem" in problem_description
            assert success_event.is_set() is False
            return {"success": True, "rounds": 1, "history": []}

    monkeypatch.setattr(
        "examples.kernel_agent_delegate.oracle.VerificationWorker",
        DummyWorker,
    )

    test_code_path = tmp_path / "test_kernel.py"
    test_code_path.write_text("print('ASSERT')\n", encoding="utf-8")
    problem_path = tmp_path / "problem.txt"
    problem_path.write_text("problem\n", encoding="utf-8")

    oracle = factory(
        {
            "mode": "kernel_worker",
            "test_code_path": str(test_code_path),
            "problem_path": str(problem_path),
            "work_root": str(tmp_path / "worker_root"),
        }
    )
    result = oracle.verify("src", "print('target')", {"unit": "unit-2"})
    assert result.passed is True
    assert getattr(DummyWorker, "ran", False) is True
