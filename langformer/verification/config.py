"""Dataclasses describing verification settings and runner options."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..constants import STRATEGY_EXACT_MATCH


def _resolve_path(
    value: Union[str, Path],
    *,
    config_root: Path,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_root / path).resolve()
    return path


@dataclass(frozen=True)
class SandboxRunnerSettings:
    """Settings specific to sandboxed verification runs."""

    run_root: Path
    timeout_s: int = 120
    isolated: bool = False
    deny_network: bool = False
    require_sentinel: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "sandbox",
            "run_root": str(self.run_root),
            "timeout_s": self.timeout_s,
            "isolated": self.isolated,
            "deny_network": self.deny_network,
            "require_sentinel": self.require_sentinel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxRunnerSettings":
        return cls(
            run_root=Path(data["run_root"]).expanduser(),
            timeout_s=int(data.get("timeout_s", 120)),
            isolated=bool(data.get("isolated", False)),
            deny_network=bool(data.get("deny_network", False)),
            require_sentinel=bool(data.get("require_sentinel", False)),
        )


@dataclass(frozen=True)
class VerificationSettings:
    """High-level verification configuration."""

    strategy: str = STRATEGY_EXACT_MATCH
    test_inputs: List[Any] = field(default_factory=list)
    sandbox: Optional[SandboxRunnerSettings] = None

    @property
    def runner_kind(self) -> str:
        return "sandbox" if self.sandbox else "plugin"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "strategy": self.strategy,
            "test_inputs": list(self.test_inputs),
        }
        if self.sandbox:
            payload["runner"] = self.sandbox.to_dict()
        else:
            payload["runner"] = {"kind": "plugin"}
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationSettings":
        runner_cfg = data.get("runner") or {}
        sandbox = None
        if runner_cfg.get("kind") == "sandbox":
            sandbox = SandboxRunnerSettings.from_dict(runner_cfg)
        return cls(
            strategy=str(data.get("strategy", STRATEGY_EXACT_MATCH)),
            test_inputs=list(data.get("test_inputs") or []),
            sandbox=sandbox,
        )


def build_verification_settings(
    config: Optional[Dict[str, Any]],
    *,
    config_root: Path,
) -> VerificationSettings:
    cfg = dict(config or {})
    runner_cfg = dict(cfg.get("runner") or {})
    sandbox = None
    if runner_cfg.get("kind") == "sandbox":
        sandbox = SandboxRunnerSettings(
            run_root=_resolve_path(
                runner_cfg.get("run_root", ".transpile_runs"),
                config_root=config_root,
            ),
            timeout_s=int(runner_cfg.get("timeout_s", 120)),
            isolated=bool(runner_cfg.get("isolated", False)),
            deny_network=bool(runner_cfg.get("deny_network", False)),
            require_sentinel=bool(runner_cfg.get("require_sentinel", False)),
        )
    return VerificationSettings(
        strategy=str(cfg.get("strategy", STRATEGY_EXACT_MATCH)),
        test_inputs=list(cfg.get("test_inputs") or []),
        sandbox=sandbox,
    )


__all__ = [
    "SandboxRunnerSettings",
    "VerificationSettings",
    "build_verification_settings",
]
