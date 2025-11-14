from pathlib import Path

from langformer.preprocessing.delegates import (
    DelegateRegistry,
    ExecutionDelegate,
    ExecutionPlan,
    load_delegate,
)


class _StubDelegate(ExecutionDelegate):
    def __init__(self, config):
        super().__init__(config)
        self.ran = False

    def execute(self, source_path, target_path, run_root, plan, config):
        self.ran = True
        target_path.write_text("delegate")
        return 0


def test_delegate_registry_round_trip(tmp_path: Path):
    registry = DelegateRegistry.get_registry()
    registry.register("stub", _StubDelegate)
    delegate = load_delegate("stub", {})
    plan = ExecutionPlan(action="delegate", context={"delegate": "stub"})
    target = tmp_path / "out.txt"
    exit_code = delegate.execute(
        tmp_path / "src.txt", target, tmp_path, plan, {}
    )
    assert exit_code == 0
    assert target.read_text() == "delegate"
