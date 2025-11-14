"""Planner base classes and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExecutionPlan:
    """Represents how the CLI should execute a request."""

    action: str = "transpile"
    context: Dict[str, Any] = field(default_factory=dict)


class ExecutionPlanner:
    """Base class for planners that can adjust execution strategy."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def plan(self, source_path: Path, config: Dict[str, Any]) -> ExecutionPlan:
        raise NotImplementedError


class NoOpPlanner(ExecutionPlanner):
    """Default planner that always runs the standard transpilation flow."""

    def plan(self, source_path: Path, config: Dict[str, Any]) -> ExecutionPlan:
        return ExecutionPlan(action="transpile", context={"reason": "default"})


class PlannerRegistry:
    """Keeps track of planner classes by name."""

    _instance: "PlannerRegistry | None" = None

    def __init__(self) -> None:
        self._registry: Dict[str | None, type[ExecutionPlanner]] = {
            None: NoOpPlanner,
            "default": NoOpPlanner,
            "noop": NoOpPlanner,
        }

    @classmethod
    def get_registry(cls) -> "PlannerRegistry":
        if cls._instance is None:
            cls._instance = PlannerRegistry()
        return cls._instance

    def register(self, name: str, planner_cls: type[ExecutionPlanner]) -> None:
        self._registry[name] = planner_cls

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

    def get(self, name: str | None) -> type[ExecutionPlanner]:
        return self._registry.get(name, NoOpPlanner)


def load_planner(
    kind: Optional[str], config: Dict[str, Any]
) -> ExecutionPlanner:
    registry = PlannerRegistry.get_registry()
    planner_cls = registry.get(kind)
    return planner_cls(config=config)


__all__ = [
    "ExecutionPlan",
    "ExecutionPlanner",
    "NoOpPlanner",
    "PlannerRegistry",
    "load_planner",
]
