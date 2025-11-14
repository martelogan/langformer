"""Example integrations for the Langformer framework."""

from __future__ import annotations

import sys

from pathlib import Path

# Ensure the vendored KernelAgent repo is importable for delegate examples.
_CANDIDATE_PATHS = [
    Path(__file__).resolve().parent / "shared_dependencies" / "KernelAgent",
    Path(__file__).resolve().parent / "shared_dependencies" / "KernelAgent",
]
for candidate in _CANDIDATE_PATHS:
    if candidate.is_dir():
        shared_str = str(candidate)
        if shared_str not in sys.path:
            sys.path.append(shared_str)
        break

# The subpackages (e.g., examples.kernel_agent_delegate) register planners/delegates when imported.
