"""Oracle registry for pluggable verification oracles."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from langformer.types import Oracle

OracleFactory = Callable[[Dict[str, Any]], Oracle]


class OracleRegistry:
    """Singleton registry for oracle factories."""

    _instance: Optional["OracleRegistry"] = None

    def __init__(self) -> None:
        self._factories: Dict[str, OracleFactory] = {}

    @classmethod
    def get_registry(cls) -> "OracleRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, factory: OracleFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str, config: Dict[str, Any]) -> Oracle:
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"No oracle registered under '{name}'")
        return factory(config)

    def get(self, name: str) -> Optional[OracleFactory]:
        return self._factories.get(name)


__all__ = ["OracleRegistry", "OracleFactory"]
