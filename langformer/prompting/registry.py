"""Registry for prompt renderers and task engines."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from langformer.prompting.backends.base import (
    PromptRenderer,
    PromptTaskEngine,
)

RendererFactory = Callable[[], PromptRenderer]
EngineFactory = Callable[[], PromptTaskEngine]

_RENDERERS: Dict[str, RendererFactory] = {}
_ENGINES: Dict[str, EngineFactory] = {}


def register_renderer(kind: str, factory: RendererFactory) -> None:
    """Register a renderer factory for the given prompt kind."""

    _RENDERERS[kind] = factory


def unregister_renderer(kind: str) -> None:
    """Remove a renderer factory for ``kind`` if it exists."""

    _RENDERERS.pop(kind, None)


def register_engine(kind: str, factory: EngineFactory) -> None:
    """Register a task-engine factory for the given prompt kind."""

    _ENGINES[kind] = factory


def unregister_engine(kind: str) -> None:
    """Remove an engine factory for ``kind`` if it exists."""

    _ENGINES.pop(kind, None)


def get_renderer(kind: str) -> Optional[PromptRenderer]:
    """Return a renderer instance for ``kind`` if registered."""

    factory = _RENDERERS.get(kind)
    return factory() if factory else None


def get_engine(kind: str) -> Optional[PromptTaskEngine]:
    """Return an engine instance for ``kind`` if registered."""

    factory = _ENGINES.get(kind)
    return factory() if factory else None


def clear_prompt_registry() -> None:
    """Remove all registered renderer and engine factories."""

    _RENDERERS.clear()
    _ENGINES.clear()
