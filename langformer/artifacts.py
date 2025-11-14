"""Helpers for managing analyzer/transpiler/verifier artifact directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langformer.configuration import ArtifactSettings
from langformer.constants import (
    ARTIFACT_STAGE_ANALYZER,
    ARTIFACT_STAGE_TRANSPILER,
    ARTIFACT_STAGE_VERIFIER,
)


@dataclass
class ArtifactManager:
    """Tracks artifact directories and manifests for each unit."""

    settings: ArtifactSettings
    _manifest: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.settings.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str, unit_id: str) -> Path:
        """Return (and create) a directory for the given stage and unit."""

        base = self._stage_root(stage) / unit_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def register(
        self,
        stage: str,
        unit_id: str,
        path: Path,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that an artifact was produced for a stage/unit."""

        record: Dict[str, Any] = {"path": str(path)}
        if metadata:
            record["metadata"] = metadata
        stage_entries = self._manifest.setdefault(unit_id, {}).setdefault(
            stage, []
        )
        stage_entries.append(record)

    def manifest_for(self, unit_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return a copy of the artifact manifest for a unit."""

        data = self._manifest.get(unit_id, {})
        return {
            stage: list(entries)
            for stage, entries in data.items()
        }

    def reset(
        self,
        unit_id: str,
        *,
        preserve: Optional[Iterable[str]] = None,
    ) -> None:
        """Drop artifacts for the unit, optionally preserving select stages."""

        if not preserve:
            self._manifest.pop(unit_id, None)
            return
        data = self._manifest.get(unit_id)
        if not data:
            return
        preserved: Dict[str, List[Dict[str, Any]]] = {}
        for stage in preserve:
            entries = data.get(stage)
            if entries:
                preserved[stage] = list(entries)
        if preserved:
            self._manifest[unit_id] = preserved
        else:
            self._manifest.pop(unit_id, None)

    def reset_all(self) -> None:
        """Clear manifests for all units."""

        self._manifest.clear()

    def _stage_root(self, stage: str) -> Path:
        if stage == ARTIFACT_STAGE_ANALYZER:
            subdir = self.settings.analyzer_dir
        elif stage == ARTIFACT_STAGE_TRANSPILER:
            subdir = self.settings.transpiler_dir
        elif stage == ARTIFACT_STAGE_VERIFIER:
            subdir = self.settings.verifier_dir
        else:
            subdir = stage
        root = (self.settings.root / subdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root


__all__ = ["ArtifactManager"]
