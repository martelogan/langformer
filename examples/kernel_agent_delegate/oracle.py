"""Oracle that reuses KernelAgent verification backends."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).
# This file has been substantially modified from its original version to integrate
# with the Langformer framework while preserving KernelAgent capabilities.

from __future__ import annotations

import multiprocessing as mp
import tempfile

from pathlib import Path
from typing import Any, Dict, Literal

from langformer.types import Oracle, VerifyResult

try:
    from Fuser.runner import run_candidate
except Exception:  # pragma: no cover - exercised when KernelAgent deps missing
    run_candidate = None  # type: ignore[assignment]

try:
    from triton_kernel_agent.worker import VerificationWorker
except Exception:  # pragma: no cover
    VerificationWorker = None  # type: ignore[assignment]

OracleMode = Literal["fuser_runner", "kernel_worker"]


def build_kernel_agent_oracle(config: Dict[str, Any]) -> Oracle:
    """Construct an oracle that runs KernelAgent verification flows."""

    cfg = config or {}
    mode: OracleMode = cfg.get("mode", "fuser_runner")
    if mode == "kernel_worker":
        return _build_kernel_worker_oracle(cfg)
    return _build_fuser_runner_oracle(cfg)


def _build_fuser_runner_oracle(config: Dict[str, Any]) -> Oracle:
    if run_candidate is None:
        raise RuntimeError(
            "Fuser.runner.run_candidate is unavailable in this environment"
        )

    run_root = Path(
        config.get("run_root", ".transpile_runs/kernel_agent_oracle")
    ).resolve()
    timeout_s = int(config.get("timeout_s", 600))
    isolated = bool(config.get("isolated", True))
    deny_network = bool(config.get("deny_network", True))
    run_root.mkdir(parents=True, exist_ok=True)

    def verify(
        source_code: str, target_code: str, metadata: Dict[str, Any]
    ) -> VerifyResult:  # noqa: ARG001
        code = target_code or ""
        with tempfile.TemporaryDirectory(
            prefix="kernel_agent_oracle_"
        ) as tmpdir:
            candidate_path = Path(tmpdir) / "candidate.py"
            candidate_path.write_text(code, encoding="utf-8")
            try:
                result = run_candidate(
                    candidate_path,
                    run_root=run_root,
                    timeout_s=timeout_s,
                    isolated=isolated,
                    deny_network=deny_network,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - defensive: surface runner error
                return VerifyResult(
                    passed=False,
                    details={
                        "strategy": "kernel_agent_runner",
                        "error": str(exc),
                        "unit": metadata.get("unit"),
                    },
                )

        details = {
            "strategy": "kernel_agent_runner",
            "unit": metadata.get("unit"),
            "run_root": str(run_root),
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "validator": result.validator_used,
            "reason": result.reason,
            "rc": result.rc,
        }
        return VerifyResult(passed=result.passed, details=details)

    return Oracle(verify=verify)


def _build_kernel_worker_oracle(config: Dict[str, Any]) -> Oracle:
    if VerificationWorker is None:
        raise RuntimeError(
            "triton_kernel_agent.worker.VerificationWorker is unavailable"
        )

    test_code_path = config.get("test_code_path")
    problem_path = config.get("problem_path")
    if not test_code_path or not problem_path:
        raise ValueError(
            "kernel_worker oracle requires 'test_code_path' and 'problem_path'"
        )

    test_code = Path(test_code_path).read_text(encoding="utf-8")
    problem_description = Path(problem_path).read_text(encoding="utf-8")
    work_root = Path(
        config.get("work_root", ".transpile_runs/kernel_worker_oracle")
    ).resolve()
    log_root = Path(config.get("log_root", work_root / "logs")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    worker_kwargs = {
        "max_rounds": int(config.get("max_rounds", 1)),
        "history_size": int(config.get("history_size", 4)),
        "openai_model": config.get("model"),
        "high_reasoning_effort": config.get("high_reasoning", True),
    }

    def verify(
        source_code: str, target_code: str, metadata: Dict[str, Any]
    ) -> VerifyResult:  # noqa: ARG001
        kernel_code = target_code or ""
        with tempfile.TemporaryDirectory(
            prefix="kernel_worker_", dir=work_root
        ) as tmpdir:
            workdir = Path(tmpdir)
            worker = VerificationWorker(
                worker_id=0,
                workdir=workdir,
                log_dir=log_root,
                **worker_kwargs,
            )
            success_event = mp.Event()
            try:
                result = worker.run(
                    kernel_code=kernel_code,
                    test_code=test_code,
                    problem_description=problem_description,
                    success_event=success_event,
                )
            except Exception as exc:  # pragma: no cover - defensive
                return VerifyResult(
                    passed=False,
                    details={
                        "strategy": "kernel_worker",
                        "error": str(exc),
                        "unit": metadata.get("unit"),
                    },
                )

        details = {
            "strategy": "kernel_worker",
            "unit": metadata.get("unit"),
            "work_root": str(work_root),
            "log_root": str(log_root),
            "rounds": result.get("rounds"),
            "history": result.get("history"),
        }
        return VerifyResult(
            passed=bool(result.get("success")), details=details
        )

    return Oracle(verify=verify)


__all__ = ["build_kernel_agent_oracle"]
