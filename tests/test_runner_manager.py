import time

from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.runtime.runner.manager import RunnerManager


def test_runner_manager_executes_plugin_code():
    manager = RunnerManager()
    plugin = LightweightPythonLanguagePlugin()
    code = """\
def main(value: int = 1):
    return value + 5
"""
    result = manager.run(plugin, code, {"value": 2})
    assert result.success
    assert result.output["result"] == 7


def test_runner_manager_times_out(monkeypatch):
    manager = RunnerManager(timeout=0.01)
    plugin = LightweightPythonLanguagePlugin()

    def slow_execute(
        code: str, inputs
    ):  # pragma: no cover - simulated slow path
        time.sleep(0.1)
        return {"result": 0}

    monkeypatch.setattr(plugin, "execute", slow_execute)
    result = manager.run(plugin, "def main():\n    return 0\n")
    assert not result.success
    assert result.error == "runner timeout"
