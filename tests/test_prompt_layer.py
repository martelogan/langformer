from __future__ import annotations

from pathlib import Path

from langformer.prompting.backends.jinja_backend import JinjaPromptRenderer
from langformer.prompting.manager import PromptManager
from langformer.prompting.types import PromptTaskSpec

PROMPT_DIR = Path("langformer/prompting/templates")


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
