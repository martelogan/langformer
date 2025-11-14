"""Language plugin registry."""

from typing import Dict, Type

from .base import LanguagePlugin
from .python import LightweightPythonLanguagePlugin
from .ruby import LightweightRubyLanguagePlugin
from .rust import LightweightRustLanguagePlugin

LANGUAGE_PLUGINS: Dict[str, Type[LanguagePlugin]] = {
    "python": LightweightPythonLanguagePlugin,
    "ruby": LightweightRubyLanguagePlugin,
    "rust": LightweightRustLanguagePlugin,
}


def register_language_plugin(
    name: str, plugin_cls: Type[LanguagePlugin]
) -> None:
    """Register or override a language plugin at runtime."""

    LANGUAGE_PLUGINS[name.lower()] = plugin_cls


def unregister_language_plugin(name: str) -> None:
    """Remove a language plugin that was previously registered."""

    LANGUAGE_PLUGINS.pop(name.lower(), None)


__all__ = [
    "LanguagePlugin",
    "LANGUAGE_PLUGINS",
    "LightweightPythonLanguagePlugin",
    "LightweightRubyLanguagePlugin",
    "LightweightRustLanguagePlugin",
    "register_language_plugin",
    "unregister_language_plugin",
]
