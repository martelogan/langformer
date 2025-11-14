# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Path utilities adapted from Fuser/paths.py."""

from __future__ import annotations

import stat

from dataclasses import dataclass
from pathlib import Path


class PathSafetyError(Exception):
    pass


def ensure_abs_regular_file(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        raise PathSafetyError(f"path must be absolute: {path}")
    try:
        st = path.lstat()
    except FileNotFoundError as exc:  # pragma: no cover
        raise PathSafetyError(f"path does not exist: {path}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise PathSafetyError(f"path must not be a symlink: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise PathSafetyError(f"path must be a regular file: {path}")
    return path


@dataclass(frozen=True)
class RunDirectories:
    """Standardized directory layout for RunSession artifacts."""

    run_dir: Path
    orchestrator: Path
    workers: Path
    digests: Path

    def __getitem__(self, key: str) -> Path:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc


def make_run_dirs(
    base: Path,
    run_id: str,
    *,
    exist_ok: bool = False,
) -> RunDirectories:
    run_dir = base / run_id
    orchestrator = run_dir / "orchestrator"
    workers = run_dir / "workers"
    digests = run_dir / "shared" / "digests"
    for directory in (run_dir, orchestrator, workers, digests):
        directory.mkdir(parents=True, exist_ok=exist_ok)
    return RunDirectories(
        run_dir=run_dir,
        orchestrator=orchestrator,
        workers=workers,
        digests=digests,
    )
