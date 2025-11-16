from __future__ import annotations

from pathlib import Path

from langformer.prompting.fills import PromptFillContext, prompt_fills
from langformer.prompting.manager import PromptManager
from langformer.types import IntegrationContext, TranspileUnit


def _base_context() -> dict:
    return {
        "source_language": "python",
        "target_language": "rust",
        "unit_kind": "module",
        "source_code": "def foo():\n    return 1\n",
        "context_overview": "{}",
        "plugin_prompt": "",
        "feedback": "",
        "attempt": 1,
        "previous_code": "",
        "guidelines": [],
        "style_targets": [],
        "quality_bar": [],
    }


def test_prompt_manager_uses_override_when_available(tmp_path: Path) -> None:
    override_dir = tmp_path / "prompts"
    override_dir.mkdir()
    override_template = override_dir / "transpile.j2"
    override_template.write_text(
        "override attempt={{ attempt }}", encoding="utf-8"
    )

    manager = PromptManager(
        Path("langformer/prompting/templates"), extra_dirs=[override_dir]
    )
    rendered = manager.render("transpile.j2", **_base_context())
    assert rendered.strip() == "override attempt=1"


def test_prompt_manager_falls_back_to_defaults(tmp_path: Path) -> None:
    override_dir = tmp_path / "only-custom"
    override_dir.mkdir()
    (override_dir / "custom.j2").write_text(
        "custom template", encoding="utf-8"
    )

    manager = PromptManager(
        Path("langformer/prompting/templates"), extra_dirs=[override_dir]
    )
    templates = manager.list_templates()
    assert "transpile.j2" in templates
    rendered = manager.render("transpile.j2", **_base_context())
    # Should contain one of the default instructions.
    assert "You are Langformer" in rendered


def test_prompt_fill_registry_merges_outputs() -> None:
    unit = TranspileUnit(
        id="demo", language="python", source_code="print('hi')"
    )
    integration_ctx = IntegrationContext(target_language="ruby")
    fill_ctx = PromptFillContext(
        unit=unit,
        integration_context=integration_ctx,
        attempt=2,
        feedback="needs work",
        previous_code=unit.source_code,
        source_language="python",
        target_language="ruby",
        source_plugin=None,
        target_plugin=None,
    )

    def _extra_fill(ctx: PromptFillContext) -> dict[str, object]:
        return {"custom_note": f"attempt_{ctx.attempt}"}

    prompt_fills.register(_extra_fill)
    try:
        payload = prompt_fills.build_payload(fill_ctx)
    finally:
        prompt_fills.unregister(_extra_fill)

    assert "snake_case" in payload["plugin_prompt"]
    assert payload["custom_note"] == "attempt_2"
    assert "guidelines" in payload
    assert "context_overview" in payload
