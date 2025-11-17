from __future__ import annotations

from pathlib import Path

import pytest

from langformer.prompting.backends.dspy_backend import BasicDSPyTranspiler
from langformer.prompting.backends.jinja_backend import JinjaPromptRenderer
from langformer.prompting.manager import PromptManager
from langformer.prompting.registry import (
    clear_prompt_registry,
    get_engine,
    get_renderer,
    register_engine,
    register_renderer,
)
from langformer.prompting.types import PromptTaskSpec

PROMPT_DIR = Path("langformer/prompting/templates")


def _require_dspy() -> None:
    pytest.importorskip(
        "dspy",
        reason="DSPy is optional; install it to run DSPy prompt-engine tests.",
    )


def _base_metadata() -> dict[str, object]:
    return {
        "source_language": "python",
        "target_language": "ruby",
        "unit_kind": "module",
        "source_code": "def main():\n    return 1\n",
        "feedback": "",
        "attempt": 1,
        "previous_code": "",
        "context_overview": "{}",
        "guidelines": ["keep behavior identical"],
        "style_targets": ["Prefer idiomatic Ruby"],
        "quality_bar": ["Return runnable Ruby code"],
    }


def test_jinja_prompt_renderer_renders_expected_template() -> None:
    manager = PromptManager(PROMPT_DIR)
    renderer = JinjaPromptRenderer(
        manager, template_map={"unit_test": "transpile.j2"}
    )
    spec = PromptTaskSpec(
        kind="unit_test",
        task_id="unit",
        metadata=_base_metadata(),
    )
    rendered = renderer.render(spec)

    assert "You are Langformer" in rendered.message.content
    assert rendered.preview == rendered.message.content
    assert rendered.template == "transpile.j2"


def test_jinja_prompt_renderer_falls_back_to_default_template() -> None:
    manager = PromptManager(PROMPT_DIR)
    renderer = JinjaPromptRenderer(manager, template_map={"exists": "refine.j2"})
    spec = PromptTaskSpec(
        kind="missing",
        task_id="unit",
        metadata=_base_metadata(),
    )
    rendered = renderer.render(spec)

    assert rendered.template == "transpile.j2"
    assert rendered.message.content


def _module_factory_with_output(text: str):
    class _StubModule:
        def __call__(self, **kwargs):
            class _Result:
                output = text

            return _Result()

    return _StubModule


def test_basic_dspy_transpiler_uses_custom_module() -> None:
    _require_dspy()
    engine = BasicDSPyTranspiler(
        module_factory=_module_factory_with_output("demo-output")
    )
    spec = PromptTaskSpec(kind="transpile_refine", task_id="demo", metadata={})

    result = engine.run(spec)

    assert result.output == "demo-output"
    assert result.metadata["source"] == "task_engine"


def test_prompt_registry_registers_renderer_and_engine() -> None:
    _require_dspy()
    clear_prompt_registry()
    manager = PromptManager(PROMPT_DIR)
    register_renderer(
        "unit_test_registry",
        lambda: JinjaPromptRenderer(
            manager, template_map={"unit_test_registry": "transpile.j2"}
        ),
    )
    register_engine(
        "unit_test_registry",
        lambda: BasicDSPyTranspiler(
            module_factory=_module_factory_with_output("registry-output")
        ),
    )
    spec = PromptTaskSpec(
        kind="unit_test_registry",
        task_id="unit",
        metadata=_base_metadata(),
    )

    renderer = get_renderer("unit_test_registry")
    engine = get_engine("unit_test_registry")

    assert renderer is not None
    assert engine is not None
    assert "Langformer" in renderer.render(spec).message.content
    assert engine.run(spec).output == "registry-output"

    clear_prompt_registry()
