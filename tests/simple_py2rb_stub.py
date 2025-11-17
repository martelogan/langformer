"""Helper for installing the deterministic Py→Rb stub during tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from examples.simple_py2rb_transpiler.outputs.test_only import (
    stub_translation,
)
from langformer.agents.transpiler import DefaultTranspilerAgent
from langformer.types import CandidatePatchSet

STUB_RUBY_IMPLEMENTATION = (
    stub_translation.STUB_RUBY_IMPLEMENTATION
)


def install_stubbed_transpiler(monkeypatch: Any) -> None:
    """Replace the LLM transpiler with a deterministic stub."""

    def _stubbed_transpile(self, unit, ctx, verifier=None, **kwargs):
        target_path = ctx.layout.target_path(unit.id)
        files = {Path(target_path): STUB_RUBY_IMPLEMENTATION}
        candidate = CandidatePatchSet(files=files, notes={"variant": "stub"})
        if verifier is not None:
            result = verifier.verify(unit, candidate, ctx)
            candidate.notes["verification"] = result.feedback.to_dict()
        return candidate

    monkeypatch.setattr(DefaultTranspilerAgent, "transpile", _stubbed_transpile)
