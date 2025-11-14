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

"""Relay provider implementation (OpenAI-compatible endpoint)."""

from __future__ import annotations

from langformer.llm.providers.openai_base import OpenAICompatibleProvider


class RelayProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        *,
        api_key_env: str = "RELAY_API_KEY",
        base_url: str = "https://relay.proxy",
    ) -> None:
        super().__init__(api_key_env=api_key_env, base_url=base_url)

    @property
    def name(self) -> str:
        return "relay"
