from __future__ import annotations

from examples.barebones.transpile_with_dspy import (
    build_demo_spec,
    register_prompt_layer_backends,
)
from langformer.prompting.registry import (
    clear_prompt_registry,
    get_engine,
    get_renderer,
)


def _stub_module_factory():
    class _Module:
        def __call__(self, **kwargs):
            class _Result:
                output = "stub-dspy-output"

            return _Result()

    return _Module


def test_transpile_with_dspy_example_registers_components() -> None:
    clear_prompt_registry()
    register_prompt_layer_backends(module_factory=_stub_module_factory())
    spec = build_demo_spec()

    renderer = get_renderer("transpile_initial")
    engine = get_engine("transpile_refine")

    assert renderer is not None
    assert engine is not None
    prompt_preview = renderer.render(spec).preview
    assert prompt_preview is not None and "Langformer" in prompt_preview
    assert engine.run(spec).output == "stub-dspy-output"

    clear_prompt_registry()
