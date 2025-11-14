"""Test plugin module that registers a planner on import."""

from __future__ import annotations

from langformer.preprocessing.planners import (
    ExecutionPlan,
    ExecutionPlanner,
    PlannerRegistry,
)


class SamplePlanner(ExecutionPlanner):
    def plan(self, source_path, config):
        return ExecutionPlan(action="delegate", context={"delegate": "sample"})


PlannerRegistry.get_registry().register("sample_plugin_planner", SamplePlanner)
