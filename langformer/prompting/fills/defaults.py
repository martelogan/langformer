"""Default prompt fill definitions."""

from __future__ import annotations

import json

from typing import Dict, Iterable, Tuple

from langformer.prompting.fills.registry import (
    PromptFillContext,
    PromptFillRegistry,
)

GUIDELINE_BASE: Tuple[str, ...] = (
    "Preserve the public API, docstrings, and side effects exactly as "
    "defined in the source module.",
    "Carry over input validation and error handling with identical "
    "semantics.",
    "Avoid introducing new external dependencies unless required by the "
    "target runtime.",
)

STYLE_TARGET_BASE: Tuple[str, ...] = (
    "Prefer idiomatic {target_language} constructs and standard libraries.",
    "Produce runnable, production-quality code—no TODOs or pseudo-code.",
)

QUALITY_BAR_BASE: Tuple[str, ...] = (
    "Semantics must match the source implementation for all supported inputs.",
    "Return only raw target-language code (no markdown fences or narration).",
    "Module must be self-contained with accurate imports and "
    "deterministic behavior.",
)

LANGUAGE_HINTS: Dict[str, list[str]] = {
    "ruby": [
        "Use snake_case methods and favor Enumerable helpers over manual "
        "loops.",
        "Return explicit values; avoid implicit printing or reliance on "
        "global state.",
        "Prefer modules/classes to namespace helpers instead of free-file "
        "globals.",
    ],
    "rust": [
        "Model shared data using structs/enums and respect "
        "ownership/borrowing rules.",
        "Use Result/Option to encode fallible operations rather than "
        "silently unwrap.",
        "Favor iterators and slices to express vectorized Python behavior "
        "idiomatically.",
    ],
}

TRANSLATION_HINTS: Dict[Tuple[str, str], list[str]] = {
    ("python", "ruby"): [
        "Convert `__init__` to `initialize` and map dataclasses to plain "
        "Ruby classes with attr_reader/attr_accessor.",
        "Replace Python `print` calls with return values unless "
        "side-effects are required.",
    ],
    ("python", "rust"): [
        "Translate Python exceptions into `Result<T, E>` or panic with the "
        "same message when failure is fatal.",
        "Mirror dynamic dictionaries with `HashMap` or typed structs and "
        "ensure lifetimes cover borrowed data.",
    ],
}


def register_target_language_hints(
    language: str, hints: Iterable[str]
) -> None:
    """Extend hints for a specific target language."""

    key = language.lower()
    values = [hint for hint in hints if hint]
    if not values:
        return
    LANGUAGE_HINTS.setdefault(key, []).extend(values)


def register_translation_hints(
    source: str, target: str, hints: Iterable[str]
) -> None:
    """Extend hints for a specific source→target translation pair."""

    key = (source.lower(), target.lower())
    values = [hint for hint in hints if hint]
    if not values:
        return
    TRANSLATION_HINTS.setdefault(key, []).extend(values)


def _guidelines_fill(ctx: PromptFillContext) -> dict[str, object]:
    feature_spec = getattr(ctx.integration_context, "feature_spec", {}) or {}
    guidelines = list(GUIDELINE_BASE)
    if feature_spec.get("pure_functions"):
        guidelines.append(
            "Maintain purity: avoid hidden I/O or shared mutable state "
            "beyond what the source performed."
        )
    style_targets = [
        entry.format(target_language=ctx.target_language)
        for entry in STYLE_TARGET_BASE
    ]
    quality_bar = list(QUALITY_BAR_BASE)
    return {
        "guidelines": guidelines,
        "style_targets": style_targets,
        "quality_bar": quality_bar,
    }


def _language_prompt_fill(ctx: PromptFillContext) -> dict[str, object]:
    hints: list[str] = []
    target_key = ctx.target_language.lower()
    hints.extend(LANGUAGE_HINTS.get(target_key, []))
    translation_key = (ctx.source_language.lower(), target_key)
    hints.extend(TRANSLATION_HINTS.get(translation_key, []))
    plugin_prompt = "\n".join(hints) if hints else ""
    return {"plugin_prompt": plugin_prompt}


def _context_snapshot_fill(ctx: PromptFillContext) -> dict[str, object]:
    snapshot = {
        "layout": ctx.integration_context.layout.as_dict(),
        "build": ctx.integration_context.build,
        "feature_spec": ctx.integration_context.feature_spec,
        "api_mappings": ctx.integration_context.api_mappings,
    }
    return {"context_overview": json.dumps(snapshot, indent=2, sort_keys=True)}


def register_default_fills(registry: PromptFillRegistry) -> None:
    registry.register(_guidelines_fill)
    registry.register(_language_prompt_fill)
    registry.register(_context_snapshot_fill)
