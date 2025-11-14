"""Expose the project root on sys.path for pytest runs."""

from __future__ import annotations

import sys

from pathlib import Path

import pytest

from langformer.agents.base import LLMConfig
from langformer.llm.providers import load_provider
from langformer.prompts.manager import PromptManager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def llm_config() -> LLMConfig:
    """Return a reusable echo-provider config for unit tests."""

    prompt_dir = Path("langformer/prompts/templates")
    prompt_manager = PromptManager(prompt_dir)
    provider = load_provider({"provider": "echo"})
    return LLMConfig(
        provider=provider,
        prompt_manager=prompt_manager,
        prompt_paths=prompt_manager.search_paths,
    )
