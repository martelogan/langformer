"""Simple Python→Ruby example that registers its custom oracle."""

from __future__ import annotations

from .oracle import register_oracle as _register_oracle

_register_oracle()

__all__ = ["_register_oracle"]
