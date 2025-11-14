"""Concrete verification strategies."""

from __future__ import annotations

import json

from typing import Any, Dict, List

from langformer.runtime.runner.manager import RunnerManager
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)
from langformer.verification.base import VerificationStrategy

from ..constants import (
    STRATEGY_CUSTOM_ORACLE,
    STRATEGY_EXACT_MATCH,
    STRATEGY_EXECUTION_MATCH,
    STRATEGY_STRUCTURAL_MATCH,
)


class ExactMatchStrategy(VerificationStrategy):
    """Verifies that source code text matches generated target text."""

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
        *,
        source_plugin,
        target_plugin,
    ) -> VerifyResult:  # noqa: ARG002 - plugins unused for exact match
        produced = _first_candidate(candidate)
        expected = unit.source_code or ""
        passed = produced.strip() == expected.strip()
        return VerifyResult(
            passed=passed,
            details={
                "expected": expected,
                "produced": produced,
                "strategy": STRATEGY_EXACT_MATCH,
            },
        )


class StructuralMatchStrategy(VerificationStrategy):
    """Placeholder structural comparison strategy."""

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
        *,
        source_plugin,
        target_plugin,
    ) -> VerifyResult:  # noqa: ARG002
        return VerifyResult(
            passed=False,
            details={
                "reason": "Structural matching not implemented",
                "strategy": STRATEGY_STRUCTURAL_MATCH,
            },
            cost={"strategy": STRATEGY_STRUCTURAL_MATCH},
        )


class ExecutionMatchStrategy(VerificationStrategy):
    """Runs both source and candidate code and compares their outputs."""

    def __init__(
        self,
        test_inputs: List[Dict[str, Any]] | None = None,
        runner_manager: RunnerManager | None = None,
    ) -> None:
        self.test_inputs = test_inputs or [{}]
        self.runner_manager = runner_manager or RunnerManager()

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
        *,
        source_plugin,
        target_plugin,
    ) -> VerifyResult:
        target_code = _first_candidate(candidate)
        inputs_series = self.test_inputs or [{}]

        try:
            source_results = [
                self.runner_manager.run(
                    source_plugin, unit.source_code or "", case
                )
                for case in inputs_series
            ]
            target_results = [
                self.runner_manager.run(target_plugin, target_code, case)
                for case in inputs_series
            ]
            if not all(r.success for r in source_results + target_results):
                errors = [
                    r.error
                    for r in source_results + target_results
                    if not r.success
                ]
                raise RuntimeError("; ".join(filter(None, errors)))
            source_outputs = [r.output for r in source_results]
            target_outputs = [r.output for r in target_results]
            passed = source_outputs == target_outputs
            details = {
                "source_outputs": _safe_json(source_outputs),
                "target_outputs": _safe_json(target_outputs),
                "cases": inputs_series,
                "strategy": STRATEGY_EXECUTION_MATCH,
            }
            return VerifyResult(passed=passed, details=details)
        except Exception as exc:  # pragma: no cover - defensive path
            return VerifyResult(
                passed=False,
                details={
                    "error": str(exc),
                    "strategy": STRATEGY_EXECUTION_MATCH,
                },
            )


class CustomOracleStrategy(VerificationStrategy):
    """Delegates verification to a user-supplied oracle."""

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
        *,
        source_plugin,
        target_plugin,
    ) -> VerifyResult:  # noqa: ARG002
        if ctx.oracle is None:
            return VerifyResult(
                passed=False,
                details={
                    "error": "No oracle configured",
                    "strategy": STRATEGY_CUSTOM_ORACLE,
                },
            )
        target_code = _first_candidate(candidate)
        oracle_result = ctx.oracle.verify(
            unit.source_code or "", target_code, {"unit": unit.id}
        )
        if isinstance(oracle_result, VerifyResult):
            return oracle_result
        details = getattr(oracle_result, "details", {})
        return VerifyResult(
            passed=oracle_result.passed,
            details=details
            if isinstance(details, dict)
            else {"details": details},
        )


def _first_candidate(candidate: CandidatePatchSet) -> str:
    if not candidate.files:
        return ""
    _, maybe_contents = next(iter(candidate.files.items()))
    return maybe_contents


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
