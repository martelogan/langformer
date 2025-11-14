# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Prompt manager backed by Jinja2 templates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from jinja2 import ChoiceLoader, Environment, FileSystemLoader, Template


class PromptManager:
    """Loads and renders named templates from one or more directories."""

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        *,
        extra_dirs: Optional[Sequence[Path]] = None,
        search_paths: Optional[Sequence[Path]] = None,
    ) -> None:
        if templates_dir is not None:
            base_dir = Path(templates_dir)
        else:
            base_dir = Path(__file__).parent / "templates"
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Templates directory not found: {base_dir}"
            )

        if search_paths:
            paths = [Path(path) for path in search_paths]
        else:
            paths = []
            if extra_dirs:
                for override in extra_dirs:
                    override_path = Path(override)
                    if not override_path.exists():
                        raise FileNotFoundError(
                            f"Prompt override directory not found: {override_path}"
                        )
                    paths.append(override_path)
            paths.append(base_dir)

        self._base_dir = base_dir
        self._search_paths = tuple(paths)
        loaders = [FileSystemLoader(str(path)) for path in self._search_paths]
        self._env = Environment(
            loader=ChoiceLoader(loaders),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context) -> str:
        template = self._get_template(template_name)
        return template.render(**context)

    def list_templates(self) -> list[str]:
        """Return the list of known template filenames."""
        return sorted(set(self._env.list_templates()))

    @property
    def templates_dir(self) -> Path:
        return self._base_dir

    @property
    def search_paths(self) -> tuple[Path, ...]:
        return self._search_paths

    def _get_template(self, template_name: str) -> Template:
        try:
            return self._env.get_template(template_name)
        except Exception as exc:  # pragma: no cover - jinja handles specifics
            raise FileNotFoundError(
                f"Template '{template_name}' not found in {self.templates_dir}"
            ) from exc
