"""Source analysis agent that partitions code into transpile units."""

from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from langformer.agents.base import LLMConfig
from langformer.languages.base import LanguagePlugin
from langformer.types import TranspileUnit


@runtime_checkable
class AnalyzerAgent(Protocol):
    """Protocol for agents that transform raw source into transpile units."""

    def analyze(
        self, source_code: str, unit_id: str, kind: str = "module"
    ) -> List[TranspileUnit]:
        """Return one or more units for downstream transpilation."""
        ...


class DefaultAnalyzerAgent:
    """Default analyzer that defers structure to the source language plugin."""

    def __init__(
        self,
        language_plugin: LanguagePlugin,
        llm_config: LLMConfig,
        *,
        config: Dict | None = None,
    ) -> None:
        self._language_plugin = language_plugin
        self._config = config or {}
        self._llm_config = llm_config
        self._artifact_manager = llm_config.artifact_manager

    def analyze(
        self, source_code: str, unit_id: str, kind: str = "module"
    ) -> List[TranspileUnit]:
        """Run parsing and optionally return multiple transpile units."""

        partitioned = self._language_plugin.partition_units(
            source_code,
            unit_id,
            config=self._config.get("analysis"),
        )
        if partitioned:
            return partitioned

        ast_tree = self._safe_parse(source_code)
        return [
            TranspileUnit(
                id=unit_id,
                language=self._language_plugin.name,
                kind=kind,
                source_code=source_code,
                source_ast=ast_tree,
                metadata={"analyzer": "basic"},
            )
        ]

    def _safe_parse(self, source_code: str):
        try:
            return self._language_plugin.parse(source_code)
        except Exception:  # pragma: no cover
            return None
