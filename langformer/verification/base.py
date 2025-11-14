"""Verification strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)


class VerificationStrategy(ABC):
    """Base strategy contract for verifying transpilation candidates."""

    @abstractmethod
    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
        *,
        source_plugin,
        target_plugin,
    ) -> VerifyResult:
        """Compare source and candidate artifacts and return verification details."""
