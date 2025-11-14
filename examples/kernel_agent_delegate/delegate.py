"""KernelAgent delegate shim that can invoke the legacy pipeline."""

from __future__ import annotations

import traceback

from pathlib import Path
from typing import Any, Dict

from langformer.preprocessing.delegates import ExecutionDelegate, ExecutionPlan
from langformer.runtime import RunSession
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
    VerifyResult,
)

try:
    from triton_kernel_agent import TritonKernelAgent
except Exception:
    TritonKernelAgent = None

try:  # pragma: no cover - exercised via integration runs
    from Fuser.pipeline import run_pipeline as _kernel_run_pipeline
except Exception:  # pragma: no cover - fallback when Fuser isn't importable
    _kernel_run_pipeline = None


class KernelAgentDelegate(ExecutionDelegate):
    """Delegate that wraps the legacy KernelAgent / Fuser pipeline."""

    _DEFAULT_PIPELINE_CFG = {
        "extract_model": "gpt-5",
        "dispatch_model": None,
        "compose_model": "o4-mini",
        "dispatch_jobs": "auto",
        "workers": 4,
        "max_iters": 5,
        "llm_timeout_s": 1200,
        "run_timeout_s": 1200,
        "verify": True,
        "compose_max_iters": 5,
    }

    _DEFAULT_KERNEL_CFG = {
        "model": "gpt-5",
        "workers": 2,
        "max_rounds": 3,
        "high_reasoning": True,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        example_cfg = (
            (config or {}).get("examples", {}).get("kernel_agent_delegate", {})
        )
        self.mode = example_cfg.get("mode", "pipeline")
        self.pipeline_cfg = example_cfg.get("pipeline", {})
        self.kernel_cfg = example_cfg.get("kernel_agent", {})

    def execute(
        self,
        source_path: Path,
        target_path: Path,
        run_root: Path,
        plan: ExecutionPlan,
        config: Dict[str, Any],
    ) -> int:
        runtime_cfg = (
            config.get("transpilation", {}).get("runtime", {})
            if config
            else {}
        )
        run_root_cfg = Path(runtime_cfg.get("run_root", run_root))
        resume_run_id = runtime_cfg.get("run_id")
        session = RunSession(
            run_root_cfg,
            run_id=resume_run_id,
            resume=bool(runtime_cfg.get("resume")),
        )
        unit_id = (
            plan.context.get("unit_id")
            or source_path.stem
            or "kernel_agent_delegate"
        )
        metadata = {
            "delegate": "kernel_agent_delegate",
            "mode": self.mode,
            "plan_context": plan.context,
        }
        session.mark_unit_started(unit_id, metadata)
        session.log_event(
            "delegate_start",
            {
                "unit_id": unit_id,
                "mode": self.mode,
                "source": str(source_path),
                "target": str(target_path),
            },
        )

        plan_context = plan.context or {}
        solver = (
            plan_context.get("solver")
            or plan_context.get("route_strategy")
            or plan_context.get("route")
        )
        kernel_cfg = self._build_kernel_config(plan_context)
        pipeline_cfg = self._build_pipeline_config(plan_context)
        try:
            if self.mode == "stub":
                result = self._run_stub(source_path, target_path)
            else:
                result = self._dispatch_solver(
                    solver,
                    source_path,
                    target_path,
                    session,
                    unit_id,
                    kernel_cfg,
                    pipeline_cfg,
                )
            session.mark_unit_completed(unit_id, "success", result)
            session.write_metadata(f"{unit_id}_delegate", result)
            session.log_event(
                "delegate_completed", {"unit_id": unit_id, "result": result}
            )
            session.write_summary(
                {"delegate": "kernel_agent_delegate", "solver": solver}
            )
            return 0
        except Exception as exc:  # pragma: no cover - exercised in integration
            tb = traceback.format_exc()
            session.log_event(
                "delegate_failed", {"unit_id": unit_id, "error": str(exc)}
            )
            session.mark_unit_completed(
                unit_id,
                "failed",
                {"error": str(exc), "traceback": tb},
            )
            session.write_summary(
                {
                    "delegate": "kernel_agent_delegate",
                    "status": "failed",
                    "solver": solver,
                }
            )
            target_path.write_text(f"# KernelAgent delegate failed\n# {exc}\n")
            return 2

    def _run_stub(
        self, source_path: Path, target_path: Path
    ) -> Dict[str, Any]:
        contents = source_path.read_text(encoding="utf-8")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            f"# delegated stub\n{contents}", encoding="utf-8"
        )
        return {
            "mode": "stub",
            "target": str(target_path),
        }

    def _dispatch_solver(
        self,
        solver: str | None,
        source_path: Path,
        target_path: Path,
        session: RunSession,
        unit_id: str,
        kernel_cfg: Dict[str, Any],
        pipeline_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        solver = (solver or "fuser").lower()
        if solver == "kernel_then_fuser":
            try:
                return self._run_kernel_agent(
                    source_path, target_path, session, unit_id, kernel_cfg
                )
            except Exception:
                return self._run_pipeline(
                    source_path, target_path, session, unit_id, pipeline_cfg
                )
        if solver == "fuser_then_kernel":
            try:
                return self._run_pipeline(
                    source_path, target_path, session, unit_id, pipeline_cfg
                )
            except Exception:
                return self._run_kernel_agent(
                    source_path, target_path, session, unit_id, kernel_cfg
                )
        if solver == "kernelagent":
            return self._run_kernel_agent(
                source_path, target_path, session, unit_id, kernel_cfg
            )
        return self._run_pipeline(
            source_path, target_path, session, unit_id, pipeline_cfg
        )

    def _run_kernel_agent(
        self,
        source_path: Path,
        target_path: Path,
        session: RunSession,
        unit_id: str,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.mode == "stub":
            return self._run_stub(source_path, target_path)
        if TritonKernelAgent is None:
            raise RuntimeError(
                "TritonKernelAgent is unavailable in this environment"
            )
        log_dir = session.dirs.workers / f"{unit_id}_kernel_agent"
        log_dir.mkdir(parents=True, exist_ok=True)
        agent = TritonKernelAgent(
            num_workers=int(cfg.get("workers", 2)),
            max_rounds=int(cfg.get("max_rounds", 3)),
            log_dir=str(log_dir),
            model_name=cfg.get("model"),
            high_reasoning_effort=cfg.get("high_reasoning", True),
        )
        try:
            result = agent.generate_kernel(
                problem_description=source_path.read_text(encoding="utf-8"),
                test_code=None,
            )
        finally:
            try:
                agent.cleanup()
            except Exception:
                pass
        if not result.get("success"):
            raise RuntimeError(
                result.get("message", "KernelAgent failed to produce a kernel")
            )
        kernel_code = result.get("kernel_code") or ""
        if not kernel_code:
            kernel_code = source_path.read_text(encoding="utf-8")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(kernel_code, encoding="utf-8")
        return {
            "mode": "kernel_agent",
            "target": str(target_path),
            "result": result,
        }

    def _run_pipeline(
        self,
        source_path: Path,
        target_path: Path,
        session: RunSession,
        unit_id: str,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if _kernel_run_pipeline is None:
            raise RuntimeError(
                "Fuser.pipeline is not available. Ensure KernelAgent modules are installed."
            )

        pipeline_root = session.dirs.run_dir / f"{unit_id}_fuser"
        pipeline_root.mkdir(parents=True, exist_ok=True)
        summary = _kernel_run_pipeline(
            problem_path=source_path.resolve(),
            extract_model=cfg["extract_model"],
            dispatch_model=cfg.get("dispatch_model"),
            compose_model=cfg["compose_model"],
            dispatch_jobs=cfg["dispatch_jobs"],
            workers=int(cfg["workers"]),
            max_iters=int(cfg["max_iters"]),
            llm_timeout_s=int(cfg["llm_timeout_s"]),
            run_timeout_s=int(cfg["run_timeout_s"]),
            out_root=pipeline_root,
            verify=bool(cfg["verify"]),
            compose_max_iters=int(cfg["compose_max_iters"]),
        )

        composition = summary.get("composition") or {}
        composed_path = composition.get("composed_path")
        if composed_path:
            composed = Path(composed_path)
            if composed.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(
                    composed.read_text(encoding="utf-8"), encoding="utf-8"
                )

        result = {
            "mode": "pipeline",
            "summary": summary,
            "target": str(target_path),
            "composed_path": composed_path,
        }
        session.write_metadata("kernel_agent_pipeline", result)
        return result

    def _build_kernel_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {**self._DEFAULT_KERNEL_CFG, **self.kernel_cfg}
        router_cfg = context.get("router_config")
        if isinstance(router_cfg, dict):
            if "ka_max_rounds" in router_cfg:
                cfg["max_rounds"] = int(router_cfg["ka_max_rounds"])
            if "ka_num_workers" in router_cfg:
                cfg["workers"] = int(router_cfg["ka_num_workers"])
            ka_model = router_cfg.get("ka_model")
            if isinstance(ka_model, str):
                cfg["model"] = ka_model
            llm_cfg = router_cfg.get("llm_models")
            if isinstance(llm_cfg, dict):
                ka_llm = llm_cfg.get("ka_model")
                if isinstance(ka_llm, str):
                    cfg["model"] = ka_llm
        return cfg

    def _build_pipeline_config(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            **self._DEFAULT_PIPELINE_CFG,
            **self.pipeline_cfg,
        }
        router_cfg = context.get("router_config")
        if isinstance(router_cfg, dict):
            if "fuser_dispatch_jobs" in router_cfg:
                cfg["dispatch_jobs"] = router_cfg["fuser_dispatch_jobs"]
            if "compose_max_iters" in router_cfg:
                cfg["compose_max_iters"] = int(router_cfg["compose_max_iters"])
            if "fuser_verify" in router_cfg:
                cfg["verify"] = bool(router_cfg["fuser_verify"])
            llm_cfg = router_cfg.get("llm_models")
            if isinstance(llm_cfg, dict):
                for key, cfg_key in (
                    ("extract", "extract_model"),
                    ("dispatch", "dispatch_model"),
                    ("compose", "compose_model"),
                ):
                    val = llm_cfg.get(key)
                    if isinstance(val, str):
                        cfg[cfg_key] = val
        return cfg

    def verify(
        self,
        unit: TranspileUnit,
        candidate: CandidatePatchSet,
        ctx: IntegrationContext,
    ) -> VerifyResult:
        if ctx.oracle is None:
            raise RuntimeError(
                "KernelAgentDelegate requires an oracle for verification"
            )
        target_code = _first_candidate(candidate)
        metadata = {
            "unit": unit.id,
            "delegate": "kernel_agent_delegate",
            "mode": self.mode,
        }
        result = ctx.oracle.verify(
            unit.source_code or "", target_code, metadata
        )
        return result


def _first_candidate(candidate: CandidatePatchSet) -> str:
    for _, contents in candidate.files.items():
        return contents
    return ""
