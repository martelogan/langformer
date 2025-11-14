"""Utilities for writing verified artifacts to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from langformer.types import CandidatePatchSet


class TargetIntegrator:
    """Writes candidate artifacts to disk and can merge multiple units."""

    def integrate(
        self, candidate: CandidatePatchSet, destination: Optional[Path] = None
    ) -> Path:
        if not candidate.files:
            raise ValueError("CandidatePatchSet contained no files to write")

        source_path, contents = next(iter(candidate.files.items()))
        output_path = (
            Path(destination) if destination is not None else Path(source_path)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(contents)
        return output_path

    def combine_candidates(
        self,
        candidates: Iterable[CandidatePatchSet],
        module_path: Path,
    ) -> CandidatePatchSet:
        snippets = []
        for candidate in candidates:
            if not candidate.files:
                continue
            _, contents = next(iter(candidate.files.items()))
            snippets.append(contents.strip())
        merged = "\n\n".join(snippets)
        return CandidatePatchSet(files={module_path: merged})
