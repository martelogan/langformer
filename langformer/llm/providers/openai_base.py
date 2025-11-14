# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI-compatible provider base (ported from KernelAgent)."""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional

from langformer.llm.providers.base import BaseProvider, LLMResponse
from langformer.llm.utils import configure_proxy_environment

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible chat APIs."""

    def __init__(
        self, api_key_env: str, base_url: Optional[str] = None
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url = base_url
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(
        self,
    ) -> None:  # pragma: no cover - exercised when SDK available
        if not OPENAI_AVAILABLE or OpenAI is None:
            return
        api_key = self._get_api_key(self.api_key_env)
        if not api_key:
            return
        self._original_proxy_env = configure_proxy_environment()
        if self.base_url:
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def get_response(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")
        params = self._build_api_params(model_name, messages, **kwargs)
        client = self.client
        if client is None:
            raise RuntimeError(f"{self.name} client missing")
        response = client.chat.completions.create(**params)  # type: ignore[attr-defined]
        logging.getLogger(__name__).debug("OpenAI response: %s", response)
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if getattr(response, "usage", None)
            else None,
        )

    def _build_api_params(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model": model_name, "messages": messages}
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)
        max_tokens_value = min(
            kwargs.get("max_tokens", 8192),
            self.get_max_tokens_limit(model_name),
        )
        if model_name.startswith(("gpt-5", "o")):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value
        if "n" in kwargs:
            params["n"] = kwargs["n"]
        if model_name.startswith("gpt-5"):
            params["reasoning_effort"] = "high"
        elif kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("o3", "o1")
        ):
            params["reasoning_effort"] = "high"
        return params

    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        return True
