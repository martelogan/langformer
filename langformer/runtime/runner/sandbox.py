# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Sandboxed runner adapted from Fuser/runner.py for Python code."""

from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import sys
import time

from pathlib import Path
from typing import Optional

from .base import ExecutionRunner, RunResult

_PY_SENTINEL = "ALL_TESTS_PASSED"


def _allowlist_env() -> dict[str, str]:
    allow: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in {"PATH", "PYTHONPATH"}:
            allow[k] = v
        elif k.startswith("LANG") or k.startswith("LC_"):
            allow[k] = v
    allow.setdefault("PYTHONHASHSEED", "0")
    allow.setdefault("OMP_NUM_THREADS", "1")
    allow.setdefault("MKL_NUM_THREADS", "1")
    allow.setdefault("OPENBLAS_NUM_THREADS", "1")
    return allow


def _write_sitecustomize_block_network(dst_dir: Path) -> None:
    code = (
        "import socket\n"
        "def _block(*a, **k):\n    raise RuntimeError('network disabled')\n"
        "class _Blocked(socket.socket):\n"
        "    def connect(self, *a, **k): _block()\n"
        "    def connect_ex(self, *a, **k): _block()\n"
        "socket.socket = _Blocked\n"
        "socket.create_connection = _block\n"
    )
    (dst_dir / "sitecustomize.py").write_text(code, encoding="utf-8")


class SandboxRunner(ExecutionRunner):
    """Runs Python code in an isolated directory with optional network bans."""

    def __init__(
        self,
        run_root: Path,
        *,
        timeout_s: int = 120,
        isolated: bool = False,
        deny_network: bool = False,
        require_sentinel: bool = False,
    ) -> None:
        self.run_root = run_root
        self.timeout_s = timeout_s
        self.isolated = isolated
        self.deny_network = deny_network
        self.require_sentinel = require_sentinel
        self.run_root.mkdir(parents=True, exist_ok=True)

    def run(
        self, plugin, code: str, inputs: Optional[dict] = None
    ) -> RunResult:
        if plugin.language_name != "python":
            return RunResult(
                False, error="SandboxRunner only supports python plugins"
            )
        run_dir = (
            self.run_root
            / f"attempt_{int(time.time() * 1000)}_{random.randint(0, 9999):04d}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        exec_file = run_dir / "candidate_main.py"
        exec_file.write_text(code, encoding="utf-8")
        if inputs:
            (run_dir / "inputs.json").write_text(
                repr(inputs), encoding="utf-8"
            )
        if self.deny_network:
            _write_sitecustomize_block_network(run_dir)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        argv = [sys.executable, "-u"]
        if self.isolated and not self.deny_network:
            argv.append("-I")
        argv.append(exec_file.name)
        env = _allowlist_env()
        with stdout_path.open("wb") as f_out, stderr_path.open("wb") as f_err:
            proc = subprocess.Popen(
                argv,
                cwd=str(run_dir),
                stdin=subprocess.DEVNULL,
                stdout=f_out,
                stderr=f_err,
                start_new_session=True,
                env=env,
            )
            try:
                rc = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                rc = -9
        stdout = (
            stdout_path.read_text(errors="replace")
            if stdout_path.exists()
            else ""
        )
        stderr = (
            stderr_path.read_text(errors="replace")
            if stderr_path.exists()
            else ""
        )
        passed = rc == 0
        if self.require_sentinel:
            passed = passed and (_PY_SENTINEL in stdout or "PASS" in stdout)
        reason = "success" if passed else f"rc={rc}"
        if stderr:
            reason += f" stderr={stderr[-200:]}"
        # Cleanup run_dir at the end for now
        shutil.rmtree(run_dir, ignore_errors=True)
        return RunResult(
            success=passed,
            output={"stdout": stdout, "stderr": stderr},
            error=reason if not passed else None,
        )
