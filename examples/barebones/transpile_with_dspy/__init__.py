"""Demonstrates registering a DSPy prompt engine with the Prompt Task Layer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from langformer.prompting.backends.dspy_backend import BasicDSPyTranspiler
from langformer.prompting.backends.jinja_backend import JinjaPromptRenderer
from langformer.prompting.manager import PromptManager
from langformer.prompting.registry import register_engine, register_renderer
from langformer.prompting.types import PromptTaskSpec


def build_demo_spec() -> PromptTaskSpec:
    """Create a PromptTaskSpec shared by the demo and tests."""

    return PromptTaskSpec(
        kind="transpile_refine",
        task_id="demo-unit",
        metadata={
            "source_language": "python",
            "target_language": "ruby",
            "unit_kind": "module",
            "source_code": "def main(x):\n    return x * 2\n",
            "feedback": "ensure code multiplies by two",
            "attempt": 1,
            "previous_code": "",
            "context_overview": "{}",
            "guidelines": ["Keep semantics identical."],
            "style_targets": ["Prefer idiomatic Ruby blocks."],
            "quality_bar": ["Return runnable Ruby code."],
        },
    )


def make_example_module_factory() -> Callable[[], object]:
    """Return a factory that builds a lightweight DSPy module."""

    import dspy

    class DemoModule(dspy.Module):
        def forward(
            self,
            source_code: str,
            target_language: str,
            **kwargs,
        ):
            # Mimic a transpilation response inline (no LLM call required).
            translated = (
                f"# Demo output for {target_language}\n"
                f"{source_code.replace('def main', 'def main')}"
            )
            return dspy.Prediction(output=translated)

    return DemoModule


def register_prompt_layer_backends(
    *,
    module_factory: Callable[[], object] | None = None,
) -> None:
    """Register renderers + DSPy engine for the demo kinds."""

    prompt_manager = PromptManager(Path("langformer/prompting/templates"))
    template_map = {
        "transpile_initial": "transpile.j2",
        "transpile_refine": "refine.j2",
    }

    def _renderer_factory() -> JinjaPromptRenderer:
        return JinjaPromptRenderer(prompt_manager, template_map=template_map)

    register_renderer("transpile_initial", _renderer_factory)
    register_renderer("transpile_refine", _renderer_factory)
    register_engine(
        "transpile_refine",
        lambda: BasicDSPyTranspiler(
            module_factory=module_factory or make_example_module_factory()
        ),
    )


__all__ = [
    "build_demo_spec",
    "make_example_module_factory",
    "register_prompt_layer_backends",
]
