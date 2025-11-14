# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Model registry for langformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type

from langformer.llm.providers.anthropic_provider import AnthropicProvider
from langformer.llm.providers.base import BaseProvider
from langformer.llm.providers.openai_provider import OpenAIProvider
from langformer.llm.providers.relay_provider import RelayProvider


@dataclass
class ModelConfig:
    name: str
    provider_class: Type[BaseProvider]
    description: str = ""


AVAILABLE_MODELS = [
    ModelConfig(
        name="o4-mini",
        provider_class=OpenAIProvider,
        description="OpenAI o-series",
    ),
    ModelConfig(
        name="gpt-5", provider_class=OpenAIProvider, description="OpenAI GPT-5"
    ),
    ModelConfig(
        name="claude-sonnet-4-20250514",
        provider_class=AnthropicProvider,
        description="Claude 4 Sonnet",
    ),
    ModelConfig(
        name="claude-opus-4-1-20250805",
        provider_class=AnthropicProvider,
        description="Claude 4 Opus",
    ),
    ModelConfig(
        name="gcp-claude-4-sonnet",
        provider_class=RelayProvider,
        description="Relay Claude 4",
    ),
]

MODEL_NAME_TO_CONFIG: Dict[str, ModelConfig] = {
    cfg.name: cfg for cfg in AVAILABLE_MODELS
}
_PROVIDER_CACHE: Dict[
    Tuple[Type[BaseProvider], Tuple[Tuple[str, Any], ...]], BaseProvider
] = {}


def _cache_key(
    provider_cls: Type[BaseProvider], kwargs: Dict[str, Any]
) -> Tuple[Type[BaseProvider], Tuple[Tuple[str, Any], ...]]:
    if not kwargs:
        return (provider_cls, tuple())
    normalized = tuple(sorted(kwargs.items()))
    return (provider_cls, normalized)


def get_model_provider(
    model_name: str, **provider_kwargs: Any
) -> BaseProvider:
    config = MODEL_NAME_TO_CONFIG.get(model_name)
    if config is None:
        available = list(MODEL_NAME_TO_CONFIG.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    provider_cls = config.provider_class
    key = _cache_key(provider_cls, provider_kwargs)
    if key not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[key] = provider_cls(**provider_kwargs)
    provider = _PROVIDER_CACHE[key]
    if not provider.is_available():
        raise ValueError(
            f"Provider '{provider.name}' for model '{model_name}' is not "
            "available."
        )
    return provider


def register_model(config: ModelConfig) -> None:
    MODEL_NAME_TO_CONFIG[config.name] = config
