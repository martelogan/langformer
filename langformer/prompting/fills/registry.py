"""Registry for prompt fill providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from langformer.languages.base import LanguagePlugin
from langformer.types import IntegrationContext, TranspileUnit


@dataclass(frozen=True)
class PromptFillContext:
    """Context made available to prompt-fill providers."""

    unit: TranspileUnit
    integration_context: IntegrationContext
    attempt: int
    feedback: Optional[str] = None
    previous_code: Optional[str] = None
    source_language: str = "unknown"
    target_language: str = "unknown"
    source_plugin: Optional[LanguagePlugin] = None
    target_plugin: Optional[LanguagePlugin] = None


FillFunc = Callable[[PromptFillContext], Dict[str, object]]


class PromptFillRegistry:
    """Stores prompt fill functions and composes their outputs."""

    def __init__(self) -> None:
        self._fills: list[FillFunc] = []

    def register(self, func: FillFunc) -> None:
        if func not in self._fills:
            self._fills.append(func)

    def unregister(self, func: FillFunc) -> None:
        if func in self._fills:
            self._fills.remove(func)

    def build_payload(self, ctx: PromptFillContext) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        for func in self._fills:
            contribution = func(ctx)
            if contribution:
                payload.update(contribution)
        return payload

    def get_registered(self) -> Iterable[FillFunc]:
        return list(self._fills)
