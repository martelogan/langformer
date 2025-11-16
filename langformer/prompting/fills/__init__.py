"""Prompt fill registry exposed for customization."""

from langformer.prompting.fills.defaults import (
    register_default_fills,
    register_target_language_hints,
    register_translation_hints,
)
from langformer.prompting.fills.registry import (
    PromptFillContext,
    PromptFillRegistry,
)

prompt_fills = PromptFillRegistry()
register_default_fills(prompt_fills)

__all__ = [
    "prompt_fills",
    "PromptFillContext",
    "register_target_language_hints",
    "register_translation_hints",
]
