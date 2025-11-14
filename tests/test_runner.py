from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.runtime.runner.plugin_runner import PluginRunner
from langformer.runtime.runner.sandbox import SandboxRunner


def test_plugin_runner_executes_code():
    runner = PluginRunner()
    plugin = LightweightPythonLanguagePlugin()
    code = """\
def main(value: int = 1):
    return value * 2
"""
    result = runner.run(plugin, code, {"value": 3})
    assert result.success
    assert result.output["result"] == 6


def test_sandbox_runner_executes_python(tmp_path):
    runner = SandboxRunner(tmp_path / "runs", timeout_s=5)
    plugin = LightweightPythonLanguagePlugin()
    code = 'print("ALL_TESTS_PASSED")\n'
    result = runner.run(plugin, code)
    assert result.success
