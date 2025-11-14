# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Factory helpers for constructing verification strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from langformer.runtime.runner.manager import RunnerManager
from langformer.runtime.runner.sandbox import SandboxRunner
from langformer.verification.base import VerificationStrategy
from langformer.verification.config import (
    VerificationSettings,
    build_verification_settings,
)
from langformer.verification.strategies import (
    CustomOracleStrategy,
    ExactMatchStrategy,
    ExecutionMatchStrategy,
    StructuralMatchStrategy,
)

from ..constants import (
    STRATEGY_CUSTOM_ORACLE,
    STRATEGY_EXACT_MATCH,
    STRATEGY_EXECUTION_MATCH,
    STRATEGY_STRUCTURAL_MATCH,
)


def build_verification_strategy(
    config: Union[VerificationSettings, Dict[str, Any]],
    *,
    source_plugin,
    target_plugin,
) -> VerificationStrategy:
    """Construct a verification strategy from typed or raw config."""

    if isinstance(config, VerificationSettings):
        settings = config
    else:
        settings = build_verification_settings(
            config,
            config_root=Path.cwd(),
        )
    runner_manager: Optional[RunnerManager] = None
    if settings.sandbox:
        sandbox = settings.sandbox
        runner_manager = RunnerManager(
            runner=SandboxRunner(
                sandbox.run_root,
                timeout_s=sandbox.timeout_s,
                isolated=sandbox.isolated,
                deny_network=sandbox.deny_network,
                require_sentinel=sandbox.require_sentinel,
            )
        )
    elif settings.strategy == STRATEGY_EXECUTION_MATCH:
        runner_manager = RunnerManager()
    if settings.strategy == STRATEGY_EXACT_MATCH:
        return ExactMatchStrategy()
    if settings.strategy == STRATEGY_STRUCTURAL_MATCH:
        return StructuralMatchStrategy()
    if settings.strategy == STRATEGY_EXECUTION_MATCH:
        return ExecutionMatchStrategy(
            test_inputs=list(settings.test_inputs),
            runner_manager=runner_manager,
        )
    if settings.strategy == STRATEGY_CUSTOM_ORACLE:
        return CustomOracleStrategy()
    raise ValueError(f"Unknown verification strategy '{settings.strategy}'")
