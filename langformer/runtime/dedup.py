# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Shared digest registration used to deduplicate worker outputs."""

from __future__ import annotations

import hashlib
import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def register_digest(
    shared_dir: Path, digest: str, worker_id: str, iter_index: int
) -> Tuple[str, Optional[str]]:
    """Atomically register a digest in ``shared_dir`` and return (status, owner)."""

    shared_dir.mkdir(parents=True, exist_ok=True)
    entry = shared_dir / f"{digest}.json"
    payload = {
        "sha256": digest,
        "owner_worker_id": worker_id,
        "iter": iter_index,
        "ts": time.time(),
    }
    try:
        with entry.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
        return "unique", None
    except FileExistsError:
        try:
            existing = json.loads(entry.read_text(encoding="utf-8"))
            owner = existing.get("owner_worker_id")
        except Exception:  # pragma: no cover - corrupted state
            owner = None
        if owner == worker_id:
            return "duplicate_same_worker", owner
        return "duplicate_cross_worker", owner


@dataclass
class CodeDeduplicator:
    """Helper that hashes code and registers digests in a shared directory."""

    shared_dir: Path
    worker_id: str

    def register(
        self, code: str, attempt_index: int
    ) -> dict[str, Optional[str]]:
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        status, owner = register_digest(
            self.shared_dir, digest, self.worker_id, attempt_index
        )
        return {"status": status, "owner": owner, "digest": digest}
