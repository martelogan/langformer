from __future__ import annotations

from typing import Any, Dict, List

from langformer.llm.providers import (
    PROVIDER_ALIASES,
    BaseProvider,
    LLMResponse,
    load_provider,
    models as model_mod,
)
from langformer.llm.providers.models import ModelConfig


def test_load_provider_echo_by_default():
    provider = load_provider({})
    output = provider.generate("hello", metadata={"source_code": "world"})
    assert output == "world"


def test_load_provider_echo_explicit():
    provider = load_provider({"provider": "echo"})
    assert provider.generate("foo") == "foo"


class _StubProvider(BaseProvider):
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.responses: List[Dict[str, Any]] = []
        super().__init__()

    def _initialize_client(self) -> None:
        self.client = object()

    def get_response(
        self, model_name: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        self.responses.append({"model": model_name, "kwargs": kwargs})
        return LLMResponse(
            content="stubbed", model=model_name, provider="stub"
        )

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "stub"


def test_load_provider_passes_options(monkeypatch):
    monkeypatch.setitem(PROVIDER_ALIASES, "stub", _StubProvider)
    provider = load_provider(
        {
            "provider": "stub",
            "model": "stub-model",
            "base_url": "http://example.com",
            "api_key_env": "CUSTOM_KEY",
            "high_reasoning_effort": True,
        }
    )
    result = provider.generate("prompt", metadata={"source_code": "ignored"})
    assert result == "stubbed"
    stub = provider._provider  # type: ignore[attr-defined]
    assert stub.init_kwargs["base_url"] == "http://example.com"
    assert stub.init_kwargs["api_key_env"] == "CUSTOM_KEY"
    assert stub.responses[0]["kwargs"]["high_reasoning_effort"] is True


def test_load_provider_model_uses_base_url(monkeypatch):
    monkeypatch.setattr(model_mod, "MODEL_NAME_TO_CONFIG", {})
    monkeypatch.setattr(model_mod, "_PROVIDER_CACHE", {})
    model_mod.MODEL_NAME_TO_CONFIG["stub-model"] = ModelConfig(
        name="stub-model",
        provider_class=_StubProvider,
        description="stub",
    )
    provider = load_provider(
        {
            "model": "stub-model",
            "base_url": "http://relay.local",
            "api_key_env": "CUSTOM_KEY",
        }
    )
    stub = provider._provider  # type: ignore[attr-defined]
    assert stub.init_kwargs["base_url"] == "http://relay.local"
    assert stub.init_kwargs["api_key_env"] == "CUSTOM_KEY"
