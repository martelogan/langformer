"""Execution planner that wraps Fuser.auto_agent routing."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from Fuser.auto_agent import (  # type: ignore
    AutoKernelRouter,
    Complexity,
    analyze_problem_code,
)

from langformer.preprocessing.planners import ExecutionPlan, ExecutionPlanner


def _complexity_to_dict(cx: Complexity) -> Dict[str, Any]:
    return {
        "has_control_flow": cx.has_control_flow,
        "has_attention_like": cx.has_attention_like,
        "has_conv_transpose": cx.has_conv_transpose,
        "has_group_norm": cx.has_group_norm,
        "has_conv": cx.has_conv,
        "pool_ops": cx.pool_ops,
        "act_ops": cx.act_ops,
        "chain_len_estimate": cx.chain_len_estimate,
        "raw_op_names": getattr(cx, "raw_op_names", {}),
    }


class KernelAgentAutoPlanner(ExecutionPlanner):
    """Thin wrapper around Fuser's autorouter heuristics and cache."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        cfg = config or {}
        self.force_delegate = bool(cfg.get("force_delegate", False))
        self.delegate_name = cfg.get("delegate_name", "kernel_agent_delegate")
        cache_path_arg = cfg.get("cache_path")
        self.cache_path = (
            Path(cache_path_arg)
            if cache_path_arg
            else Path(".fuse") / "router_cache.json"
        )
        self.router_cfg = cfg.get("router") or {}
        self.router_enabled = bool(self.router_cfg.get("model"))
        self._router: Optional[AutoKernelRouter] = None

    def plan(self, source_path: Path, config: Dict[str, Any]) -> ExecutionPlan:
        try:
            code = source_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ExecutionPlan(
                action="transpile", context={"error": "missing_source"}
            )

        cache = self._read_cache()
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        cached_entry = cache.get(digest)

        context: Dict[str, Any] = {
            "planner": "kernel_agent_auto",
            "source": str(source_path),
        }

        route_cfg: Dict[str, Any] = {}

        if cached_entry:
            strategy = cached_entry.get("route_strategy")
            confidence = cached_entry.get("confidence")
            complexity_data = cached_entry.get("complexity")
            route_source = "cache"
            route = cached_entry.get(
                "route",
                self._route_from_strategy(strategy, default="kernel_agent"),
            )
            entry_cfg = cached_entry.get("config")
            if isinstance(entry_cfg, dict):
                route_cfg = entry_cfg
        else:
            complexity = analyze_problem_code(code)
            heuristic_prefers_fuser = complexity.route_to_fuser()
            strategy = None
            confidence = None
            route_source = "analysis"
            if self.router_enabled:
                strategy, confidence, route_cfg = self._decide_with_router(
                    source_path, code, complexity
                )
                if strategy:
                    route_source = "router"
            strategy = self._normalize_strategy(
                strategy, heuristic_prefers_fuser
            )
            route = self._route_from_strategy(
                strategy, heuristic_prefers_fuser
            )

            complexity_data = _complexity_to_dict(complexity)
            cache[digest] = {
                "route": route,
                "unit": source_path.name,
                "complexity": complexity_data,
                "route_strategy": strategy,
                "confidence": confidence,
                "config": route_cfg,
            }
            self._write_cache(cache)

        context["route"] = route
        context["route_source"] = route_source
        if strategy:
            context["route_strategy"] = strategy
        if complexity_data:
            context["complexity"] = complexity_data
        if confidence is not None:
            context["router_confidence"] = confidence

        solver = self._solver_from_strategy(strategy, route)
        if route_cfg:
            context["router_config"] = route_cfg
        context["solver"] = solver
        return ExecutionPlan(
            action="delegate",
            context={**context, "delegate": self.delegate_name},
        )

    def _decide_with_router(
        self,
        source_path: Path,
        code: str,
        complexity: Complexity,
    ) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
        router = self._ensure_router()
        if router is None:
            return None, None, {}
        try:
            strategy, confidence, info = router._llm_decide_route(  # type: ignore[attr-defined]
                source_path, code, complexity
            )
            parsed = info.get("parsed") if isinstance(info, dict) else None
            cfg: Dict[str, Any] = {}
            if isinstance(parsed, dict):
                route_cfg = parsed.get("config")
                if isinstance(route_cfg, dict):
                    cfg = route_cfg
            return strategy, confidence, cfg
        except Exception:
            return None, None, {}

    def _ensure_router(self) -> Optional[AutoKernelRouter]:
        if not self.router_enabled:
            return None
        if self._router is not None:
            return self._router
        router_args = {
            "ka_model": None,
            "ka_num_workers": 1,
            "ka_max_rounds": 1,
            "ka_high_reasoning": True,
            "router_model": self.router_cfg.get("model"),
            "router_high_reasoning": self.router_cfg.get(
                "high_reasoning", True
            ),
            "router_temperature": self.router_cfg.get("temperature", 0.2),
            "router_max_tokens": int(self.router_cfg.get("max_tokens", 700)),
            "extract_model": self.router_cfg.get("extract_model", "gpt-5"),
            "dispatch_model": self.router_cfg.get("dispatch_model", "o4-mini"),
            "compose_model": self.router_cfg.get("compose_model", "o4-mini"),
            "workers": 1,
            "max_iters": 1,
            "llm_timeout_s": 120,
            "run_timeout_s": 120,
            "compose_max_iters": 1,
            "verify": False,
            "dispatch_jobs": 1,
            "allow_fallback": False,
        }
        try:
            self._router = AutoKernelRouter(**router_args)
        except Exception:
            self._router = None
        return self._router

    def _read_cache(self) -> Dict[str, Any]:
        if not self.cache_path.is_file():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_cache(self, cache: Dict[str, Any]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(cache, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _normalize_strategy(
        self, strategy: Optional[str], heuristic_prefers_fuser: bool
    ) -> str:
        valid = {
            "kernelagent",
            "fuser",
            "kernel_then_fuser",
            "fuser_then_kernel",
        }
        if strategy and strategy in valid:
            return strategy
        return "fuser" if heuristic_prefers_fuser else "kernelagent"

    def _route_from_strategy(
        self,
        strategy: Optional[str],
        heuristic_prefers_fuser: Optional[bool] = None,
        default: str = "kernel_agent",
    ) -> str:
        if strategy in ("kernelagent", "kernel_then_fuser"):
            return "kernel_agent"
        if strategy in ("fuser", "fuser_then_kernel"):
            return "fuser"
        if heuristic_prefers_fuser is not None:
            return "fuser" if heuristic_prefers_fuser else "kernel_agent"
        return default

    def _solver_from_strategy(
        self, strategy: Optional[str], route: str
    ) -> str:
        valid = {
            "kernelagent",
            "fuser",
            "kernel_then_fuser",
            "fuser_then_kernel",
        }
        if strategy in valid:
            return strategy
        return "fuser" if route == "fuser" else "kernelagent"


__all__ = ["KernelAgentAutoPlanner"]
