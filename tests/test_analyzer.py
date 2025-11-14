import textwrap

from langformer.agents.analyzer import DefaultAnalyzerAgent
from langformer.languages.python import LightweightPythonLanguagePlugin


def test_analyzer_splits_python_functions(llm_config):
    plugin = LightweightPythonLanguagePlugin()
    analyzer = DefaultAnalyzerAgent(
        plugin, llm_config, config={"analysis": {"split_functions": True}}
    )
    source = textwrap.dedent(
        """
        def first(x):
            return x + 1

        def second(y):
            return y * 2
        """
    )

    units = analyzer.analyze(source, unit_id="module")

    assert len(units) == 2
    ids = {unit.id for unit in units}
    assert "module:first" in ids
    assert "module:second" in ids
