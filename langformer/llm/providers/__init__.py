# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""LLM provider registry."""

from __future__ import annotations

import logging

from typing import Any, Dict, Optional, Protocol

from langformer.llm.providers.anthropic_provider import AnthropicProvider
from langformer.llm.providers.base import BaseProvider, LLMResponse
from langformer.llm.providers.models import get_model_provider
from langformer.llm.providers.openai_provider import OpenAIProvider
from langformer.llm.providers.relay_provider import RelayProvider


class EchoProvider:
    """Deterministic provider for testing."""

    def generate(
        self, prompt: str, **kwargs
    ) -> str:  # pragma: no cover - trivial
        metadata = kwargs.get("metadata") or {}
        return metadata.get("source_code") or prompt


PROVIDER_ALIASES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "relay": RelayProvider,
}


_LOGGER = logging.getLogger(__name__)


def load_provider(config: Dict[str, Any]) -> LLMProvider:
    """Load a provider from config.

    Expected keys:
      - provider: optional explicit provider name (e.g., "openai", "echo")
      - model: optional model name mapped via models registry
    """

    provider_name = config.get("provider")
    model_name: Optional[str] = config.get("model")

    if provider_name == "echo" or (
        provider_name is None and model_name is None
    ):
        _LOGGER.warning(
            "LLM provider is set to 'echo'; generation will simply mirror prompts. "
            "Update `transpilation.llm.provider` / `model` to use a real LLM."
        )
        return EchoProvider()

    default_kwargs: Dict[str, Any] = {}
    for key in ("high_reasoning_effort", "max_tokens", "temperature"):
        if key in config:
            default_kwargs[key] = config[key]

    provider_kwargs: Dict[str, Any] = {}
    if config.get("base_url"):
        provider_kwargs["base_url"] = config["base_url"]
    if config.get("api_key_env"):
        provider_kwargs["api_key_env"] = config["api_key_env"]

    if provider_name and provider_name in PROVIDER_ALIASES:
        provider_cls = PROVIDER_ALIASES[provider_name]
        provider = provider_cls(**provider_kwargs)
        if not provider.is_available():  # pragma: no cover
            raise ValueError(f"Provider '{provider_name}' is not available")
        model = model_name or config.get("model", provider_name)
        _LOGGER.info(
            "Using LLM provider '%s' with model '%s'", provider_name, model
        )
        return ProviderAdapter(provider, model, default_kwargs=default_kwargs)

    if model_name:
        provider = get_model_provider(model_name, **provider_kwargs)
        _LOGGER.info("Using model '%s' via registry provider", model_name)
        return ProviderAdapter(
            provider, model_name, default_kwargs=default_kwargs
        )

    raise ValueError("llm config must specify either 'provider' or 'model'")


__all__ = [
    "BaseProvider",
    "LLMResponse",
    "LLMProvider",
    "load_provider",
]


class LLMProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...


class ProviderAdapter(LLMProvider):
    def __init__(
        self,
        provider: BaseProvider,
        model_name: str,
        *,
        default_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._default_kwargs = default_kwargs or {}

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        merged_kwargs = {**self._default_kwargs, **kwargs}
        extra = dict(merged_kwargs)
        metadata = extra.pop("metadata", None)
        response = self._provider.get_response(
            self._model_name,
            messages,
            metadata=metadata,
            **extra,
        )
        return response.content or ""
