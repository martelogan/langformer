import shutil

import pytest

from langformer.languages import (
    LANGUAGE_PLUGINS,
    LanguagePlugin,
    register_language_plugin,
    unregister_language_plugin,
)
from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.languages.ruby import LightweightRubyLanguagePlugin
from langformer.languages.rust import LightweightRustLanguagePlugin


def test_python_plugin_executes_main_callable():
    plugin = LightweightPythonLanguagePlugin()
    code = """\
def main(value: int = 3):
    return value * 2
"""
    result = plugin.execute(code, {"value": 5})
    assert result["result"] == 10


@pytest.mark.skipif(
    shutil.which("rustc") is None, reason="rustc not available"
)
def test_rust_plugin_runs_binary(tmp_path):
    plugin = LightweightRustLanguagePlugin()
    rust_code = """\
fn main() {
    println!("42");
}
"""
    output = plugin.execute(rust_code)
    assert "42" in output["stdout"].strip()


@pytest.mark.skipif(shutil.which("ruby") is None, reason="ruby not available")
def test_ruby_plugin_executes_script():
    plugin = LightweightRubyLanguagePlugin()
    ruby_code = """\
def main()
  puts 21 + 21
end

main
"""
    output = plugin.execute(ruby_code)
    assert output["stdout"].strip() == "42"


def test_register_language_plugin_round_trip():
    class DummyPlugin(LanguagePlugin):
        language_name = "dummy"

        def parse(self, source_code: str) -> str:
            return source_code

        def compile(self, code: str) -> bool:
            return True

        def execute(self, code: str, inputs=None):
            return {"echo": code}

    register_language_plugin("dummy", DummyPlugin)
    try:
        assert LANGUAGE_PLUGINS["dummy"] is DummyPlugin
    finally:
        unregister_language_plugin("dummy")
