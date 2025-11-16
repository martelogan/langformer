"""PromptRenderer backed by the existing Jinja PromptManager."""

from __future__ import annotations

from typing import Mapping, MutableMapping

from langformer.prompting.backends.base import PromptRenderer
from langformer.prompting.manager import PromptManager
from langformer.prompting.types import (
    ChatMessage,
    PromptTaskSpec,
    RenderedPrompt,
)


class JinjaPromptRenderer(PromptRenderer):
    """Adapter that renders PromptTaskSpecs via Langformer's PromptManager."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        template_map: Mapping[str, str] | None = None,
        default_template: str = "transpile.j2",
    ) -> None:
        self._prompt_manager = prompt_manager
        self._template_map: MutableMapping[str, str] = dict(template_map or {})
        self._default_template = default_template

    def render(self, spec: PromptTaskSpec) -> RenderedPrompt:
        template_name = self._template_map.get(spec.kind, self._default_template)
        # Pass metadata items directly into the template context.
        text = self._prompt_manager.render(template_name, **spec.metadata)
        message = ChatMessage(role="user", content=text)
        return RenderedPrompt(
            message=message,
            preview=text,
            template=template_name,
        )
