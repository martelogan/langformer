# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Utility helpers for LLM providers (ported from KernelAgent)."""

from __future__ import annotations

import os

from typing import Dict, Optional


def configure_proxy_environment() -> Optional[Dict[str, Optional[str]]]:
    """Apply LANGFORMER_PROXY_OVERRIDE (if set) to the common proxy env vars."""
    original: Dict[str, Optional[str]] = {}
    proxy = os.environ.get("LANGFORMER_PROXY_OVERRIDE")
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        original[key] = os.environ.get(key)
        if proxy:
            os.environ[key] = proxy
    return original
