# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path

import pytest

import langformer

from langformer import TranspilationOrchestrator
from langformer.agents.transpiler import LLMTranspilerAgent
from langformer.languages.python import LightweightPythonLanguagePlugin
from langformer.verification.strategies import (
    ExactMatchStrategy,
    ExecutionMatchStrategy,
)


def test_package_imports():
    """Ensure the new langformer package exposes orchestrator."""
    assert hasattr(langformer, "TranspilationOrchestrator")


def test_orchestrator_transpile_file(tmp_path: Path):
    """Transpile a sample file using the scaffolded orchestrator."""
    orchestrator = TranspilationOrchestrator(
        config={
            "transpilation": {
                "source_language": "python",
                "target_language": "python",
                "layout": {
                    "output": {"path": "transpiled_output.py", "kind": "file"}
                },
                "analysis": {"split_functions": True},
                "agents": {"max_retries": 2},
                "verification": {"strategy": "exact_match"},
                "llm": {"provider": "echo", "model": "mock"},
            }
        },
        source_plugin=LightweightPythonLanguagePlugin(),
        target_plugin=LightweightPythonLanguagePlugin(),
        verification_strategy=ExactMatchStrategy(),
    )

    source_path = tmp_path / "sample.py"
    target_path = tmp_path / "output.py"
    source_code = (
        "def add(a: int = 1, b: int = 2):\n    return a + b\n\n"
        "def subtract(a: int = 2, b: int = 1):\n    return a - b\n"
    )
    source_path.write_text(source_code)

    orchestrator.transpile_file(source_path, target_path)
    contents = target_path.read_text()
    assert "def add" in contents
    assert "def subtract" in contents


def test_missing_source_file_raises(tmp_path: Path):
    orchestrator = TranspilationOrchestrator(
        config={
            "transpilation": {
                "source_language": "python",
                "target_language": "python",
                "layout": {
                    "output": {"path": "transpiled_output.py", "kind": "file"}
                },
                "verification": {"strategy": "execution_match"},
                "llm": {"provider": "echo", "model": "mock"},
            }
        },
        source_plugin=LightweightPythonLanguagePlugin(),
        target_plugin=LightweightPythonLanguagePlugin(),
        verification_strategy=ExecutionMatchStrategy(),
    )

    with pytest.raises(FileNotFoundError):
        orchestrator.transpile_file(
            tmp_path / "missing.py", tmp_path / "out.py"
        )


def test_orchestrator_uses_llm_agent_by_default():
    orchestrator = TranspilationOrchestrator(
        config={
            "transpilation": {
                "source_language": "python",
                "target_language": "python",
                "layout": {
                    "output": {"path": "transpiled_output.py", "kind": "file"}
                },
                "agents": {"max_retries": 1},
                "llm": {"provider": "echo"},
            }
        },
        source_plugin=LightweightPythonLanguagePlugin(),
        target_plugin=LightweightPythonLanguagePlugin(),
        verification_strategy=ExactMatchStrategy(),
    )

    assert isinstance(orchestrator.transpiler_agent, LLMTranspilerAgent)


def test_orchestrator_worker_manager(tmp_path: Path):
    orchestrator = TranspilationOrchestrator(
        config={
            "transpilation": {
                "source_language": "python",
                "target_language": "python",
                "layout": {
                    "output": {"path": "transpiled_output.py", "kind": "file"}
                },
                "agents": {
                    "worker_manager": {
                        "enabled": True,
                        "workers": 1,
                        "variants_per_unit": 1,
                    },
                },
                "verification": {"strategy": "exact_match"},
                "llm": {"provider": "echo"},
            }
        },
    )
    source_path = tmp_path / "sample.py"
    target_path = tmp_path / "output.py"
    source_path.write_text("def main():\n    return 1\n")
    orchestrator.transpile_file(source_path, target_path)
    assert target_path.exists()


def test_orchestrator_streaming_writes_console_log(
    tmp_path: Path, monkeypatch
):
    streams_root = tmp_path / "streams"

    class _StubAdapter:
        def __init__(
            self,
            model,
            store_responses,
            timeout_s,
            jsonl_path,
            on_delta=None,
            **kwargs,
        ):
            self.jsonl_path = Path(jsonl_path)
            self.on_delta = on_delta

        def stream(self, **kwargs):
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.write_text("[]", encoding="utf-8")
            if self.on_delta:
                self.on_delta("piece-1")
                self.on_delta("piece-2")
            return {"output_text": "def main():\n    return 1\n"}

    monkeypatch.setattr("langformer.orchestrator.EventAdapter", _StubAdapter)

    orchestrator = TranspilationOrchestrator(
        config={
            "transpilation": {
                "source_language": "python",
                "target_language": "python",
                "layout": {
                    "output": {"path": "transpiled_output.py", "kind": "file"}
                },
                "llm": {
                    "provider": "echo",
                    "streaming": {
                        "enabled": True,
                        "log_root": str(streams_root),
                        "mode": "file",
                    },
                },
            }
        }
    )

    orchestrator.transpile_code("def main():\n    return 1\n", unit_id="unit")

    console_log = (
        streams_root / "orchestrator" / "unit" / "orchestrator" / "console.log"
    )
    assert console_log.exists()
    assert "piece-1" in console_log.read_text()
