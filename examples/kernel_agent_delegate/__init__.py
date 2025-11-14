"""Example components for integrating legacy KernelAgent flows."""

from __future__ import annotations

from langformer.preprocessing.delegates import DelegateRegistry
from langformer.preprocessing.planners import PlannerRegistry
from langformer.verification.oracles import OracleRegistry

from .delegate import KernelAgentDelegate
from .oracle import build_kernel_agent_oracle
from .planner import KernelAgentAutoPlanner

PlannerRegistry.get_registry().register(
    "kernel_agent_auto", KernelAgentAutoPlanner
)
DelegateRegistry.get_registry().register(
    "kernel_agent_delegate", KernelAgentDelegate
)
OracleRegistry.get_registry().register(
    "kernel_agent_runner", build_kernel_agent_oracle
)

__all__ = [
    "KernelAgentAutoPlanner",
    "KernelAgentDelegate",
    "build_kernel_agent_oracle",
]
