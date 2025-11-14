"""Shared agent primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langformer.llm.providers import LLMProvider
from langformer.prompts.manager import PromptManager

if TYPE_CHECKING:  # pragma: no cover
    from langformer.artifacts import ArtifactManager


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Container describing how agents access prompts and providers."""

    provider: LLMProvider
    prompt_manager: PromptManager
    prompt_paths: tuple[Path, ...]
    artifact_manager: "ArtifactManager | None" = None
