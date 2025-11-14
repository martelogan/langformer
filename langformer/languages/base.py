"""Base language plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from langformer.types import TranspileUnit


class LanguagePlugin(ABC):
    """Shared contract for language-specific behaviors."""

    language_name: str = "unknown"

    @property
    def name(self) -> str:
        return self.language_name

    @abstractmethod
    def parse(self, source_code: str) -> Any:
        """Return an AST or intermediate data structure."""

    @abstractmethod
    def compile(self, code: str) -> bool:
        """Check whether code compiles or is syntactically valid."""

    @abstractmethod
    def execute(
        self, code: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute code using provided inputs and return results."""

    def partition_units(
        self,
        source_code: str,
        unit_id: str,
        *,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[List["TranspileUnit"]]:
        """Optional hook for splitting source code into multiple transpile units."""
        return None
