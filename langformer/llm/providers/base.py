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

"""Base provider interfaces ported from KernelAgent."""

from __future__ import annotations

import os

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self) -> None:
        self.client: Any | None = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the provider's client."""

    @abstractmethod
    def get_response(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a single response."""

    def get_multiple_responses(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        n: int = 1,
        **kwargs: Any,
    ) -> List[LLMResponse]:
        """Return multiple responses (may be emulated if provider lacks native n>1)."""
        return [
            self.get_response(model_name, messages, **kwargs) for _ in range(n)
        ]

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the provider can be used (API key present, SDK installed, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""

    def supports_multiple_completions(self) -> bool:
        return False

    def get_max_tokens_limit(self, model_name: str) -> int:
        return 8192

    def _get_api_key(self, env_var: str) -> Optional[str]:
        api_key = os.getenv(env_var)
        if api_key and api_key != "your-api-key-here":
            return api_key
        return None
