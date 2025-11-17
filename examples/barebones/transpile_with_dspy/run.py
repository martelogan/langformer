"""Run the DSPy prompt engine demo."""

from __future__ import annotations

import sys

from pathlib import Path


def main() -> None:
    """Register the demo backends and print sample outputs."""

    _ensure_project_root()
    from examples.barebones.transpile_with_dspy import (
        build_demo_spec,
        register_prompt_layer_backends,
    )
    from langformer.prompting.registry import (
        clear_prompt_registry,
        get_engine,
        get_renderer,
    )

    clear_prompt_registry()
    register_prompt_layer_backends()
    spec = build_demo_spec()

    renderer = get_renderer("transpile_initial")
    if renderer:
        rendered = renderer.render(spec)
        print("Rendered prompt preview:\n")
        print(rendered.preview or rendered.message.content)
    else:  # pragma: no cover - demo guard
        print("No renderer registered for transpile_initial.")

    engine = get_engine("transpile_refine")
    if engine:
        result = engine.run(spec)
        print("\nDSPy engine output:\n")
        print(result.output)
    else:  # pragma: no cover - demo guard
        print("No DSPy engine registered for transpile_refine.")


def _ensure_project_root() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
