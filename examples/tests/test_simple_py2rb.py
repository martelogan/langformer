from __future__ import annotations

import shutil

from pathlib import Path

import pytest
import yaml

from langformer import TranspilationOrchestrator
from tests.simple_py2rb_stub import install_stubbed_transpiler


@pytest.mark.skipif(
    shutil.which("ruby") is None,
    reason="Ruby interpreter is required for this example",
)
def test_simple_py2rb_transpiler(monkeypatch, tmp_path: Path):
    """Run the example with a stubbed LLM output to exercise the pipeline."""

    install_stubbed_transpiler(monkeypatch)

    config = yaml.safe_load(
        Path("configs/simple_py2rb.yaml").read_text()
    )
    config["transpilation"]["llm"] = {"provider": "echo"}
    plugin_cfg = config["transpilation"].setdefault("plugins", {})
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples"
    plugin_cfg["paths"] = [str(examples_dir.resolve())]

    orchestrator = TranspilationOrchestrator(config=config)
    target = tmp_path / "sample_module.rb"
    orchestrator.transpile_file(
        Path("examples/simple_py2rb_transpiler/inputs/sample_module.py"),
        target,
        verify=True,
    )

    assert target.exists()
    contents = target.read_text()
    assert "module Report" in contents
