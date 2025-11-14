"""Verification agent wiring strategies to orchestrator flow."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from langformer.agents.base import LLMConfig
from langformer.languages.base import LanguagePlugin
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)
from langformer.verification.base import VerificationStrategy


@runtime_checkable
class VerifierAgent(Protocol):
    """Protocol for verifiers that assess candidate outputs."""

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
    ) -> VerifyResult:
        """Return verification result using configured strategy."""
        ...


class DefaultVerificationAgent:
    """Runs configured verification strategies."""

    def __init__(
        self,
        strategy: Optional[VerificationStrategy] = None,
        *,
        source_plugin: LanguagePlugin,
        target_plugin: LanguagePlugin,
        llm_config: LLMConfig,
    ) -> None:
        self._strategy = strategy
        self._source_plugin = source_plugin
        self._target_plugin = target_plugin
        self._llm_config = llm_config
        self._artifact_manager = llm_config.artifact_manager

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
    ) -> VerifyResult:
        """Use configured strategy or return a default pass."""
        if self._strategy is None:
            return VerifyResult(
                passed=True, details={"reason": "verification disabled"}
            )
        return self._strategy.verify(
            unit,
            candidate,
            ctx,
            source_plugin=self._source_plugin,
            target_plugin=self._target_plugin,
        )
