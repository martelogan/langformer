"""Delegate abstractions for preprocessing / routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)

from .planners import ExecutionPlan


class ExecutionDelegate:
    """Invoked when a plan wants to hand off execution."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def execute(
        self,
        source_path: Path,
        target_path: Path,
        run_root: Path,
        plan: ExecutionPlan,
        config: Dict[str, Any],
    ) -> int:
        raise NotImplementedError

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
    ) -> VerifyResult:
        raise NotImplementedError("Delegate did not implement verify()")


class NoOpDelegate(ExecutionDelegate):
    """Default delegate that simply aborts."""

    def execute(
        self,
        source_path: Path,
        target_path: Path,
        run_root: Path,
        plan: ExecutionPlan,
        config: Dict[str, Any],
    ) -> int:
        raise RuntimeError("No delegate registered for this plan")


class DelegateRegistry:
    """Keeps track of delegates by name."""

    _instance: "DelegateRegistry | None" = None

    def __init__(self) -> None:
        self._registry: Dict[str | None, type[ExecutionDelegate]] = {
            None: NoOpDelegate,
            "default": NoOpDelegate,
        }

    @classmethod
    def get_registry(cls) -> "DelegateRegistry":
        if cls._instance is None:
            cls._instance = DelegateRegistry()
        return cls._instance

    def register(
        self, name: str, delegate_cls: type[ExecutionDelegate]
    ) -> None:
        self._registry[name] = delegate_cls

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

    def get(self, name: Optional[str]) -> type[ExecutionDelegate]:
        return self._registry.get(name, NoOpDelegate)


def load_delegate(
    name: Optional[str], config: Dict[str, Any]
) -> ExecutionDelegate:
    registry = DelegateRegistry.get_registry()
    delegate_cls = registry.get(name)
    return delegate_cls(config=config)


__all__ = [
    "ExecutionDelegate",
    "ExecutionPlan",
    "NoOpDelegate",
    "DelegateRegistry",
    "load_delegate",
]
