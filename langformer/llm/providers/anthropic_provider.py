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

"""Anthropic provider implementation."""

from __future__ import annotations

from typing import Any, Dict, List

from langformer.llm.providers.base import BaseProvider, LLMResponse

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    ANTHROPIC_AVAILABLE = False
    anthropic = None  # type: ignore


class AnthropicProvider(BaseProvider):
    def __init__(self, *, api_key_env: str = "ANTHROPIC_API_KEY") -> None:
        self._api_key_env = api_key_env
        super().__init__()

    def _initialize_client(self) -> None:
        if not ANTHROPIC_AVAILABLE or anthropic is None:
            self.client = None
            return
        api_key = self._get_api_key(self._api_key_env)
        if not api_key:
            self.client = None
            return
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None

    def get_response(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")

        # Anthropic expects separate system prompt and conversation
        system_prompt = ""
        converted_messages = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                system_prompt = message.get("content", "")
                continue
            converted_messages.append(
                {"role": role, "content": message.get("content", "")}
            )

        client = self.client
        if client is None:
            raise RuntimeError("Anthropic client not initialized")
        response: Any = client.messages.create(  # type: ignore[attr-defined]
            model=model_name,
            system=system_prompt,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in converted_messages
            ],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=min(
                kwargs.get("max_tokens", 4096),
                self.get_max_tokens_limit(model_name),
            ),
        )
        content = ""
        blocks = getattr(response, "content", None)
        if isinstance(blocks, list) and blocks:
            first_block = blocks[0]
            text_value = getattr(first_block, "text", None)
            if isinstance(text_value, str):
                content = text_value
        return LLMResponse(
            content=content, model=model_name, provider=self.name
        )
