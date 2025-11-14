"""Utilities for constructing IntegrationContext objects."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langformer.artifacts import ArtifactManager
from langformer.configuration import IntegrationSettings
from langformer.types import (
    IntegrationContext,
    LayoutPlan,
    Oracle,
    VerifyResult,
)
from langformer.verification.oracles import OracleRegistry

from ..constants import STRATEGY_CUSTOM_ORACLE


class ContextBuilder:
    """Build IntegrationContext objects from typed settings."""

    def build(
        self,
        integration: IntegrationSettings,
        layout: LayoutPlan,
        artifacts: Optional[ArtifactManager] = None,
    ) -> IntegrationContext:
        oracle_cfg = integration.oracle

        return IntegrationContext(
            target_language=integration.target_language,
            runtime_adapter=integration.runtime_adapter,
            contract=integration.contract,
            layout=layout,
            build=integration.build,
            oracle=self._build_oracle(oracle_cfg) if oracle_cfg else None,
            api_mappings=integration.api_mappings,
            feature_spec=integration.feature_spec,
            artifacts=artifacts,
        )

    def _build_oracle(self, oracle_cfg: Dict[str, Any]) -> Oracle:
        oracle_type = oracle_cfg.get("type")
        if not oracle_type:
            raise ValueError("Oracle configuration missing 'type'")

        registry = OracleRegistry.get_registry()
        factory = registry.get(str(oracle_type))
        if factory:
            return factory(oracle_cfg)

        if oracle_type == "contains_keyword":
            keyword = oracle_cfg.get("keyword")
            if not keyword:
                raise ValueError("contains_keyword oracle requires 'keyword'")

            def verify(
                source_code: str, target_code: str, metadata: Dict[str, Any]
            ) -> VerifyResult:
                passed = keyword in target_code
                return VerifyResult(
                    passed=passed,
                    details={
                        "strategy": STRATEGY_CUSTOM_ORACLE,
                        "keyword": keyword,
                        "unit": metadata.get("unit"),
                    },
                )

            return Oracle(verify=verify)

        raise ValueError(f"Unsupported oracle type '{oracle_type}'")
