"""Typed payloads exchanged between the orchestrator and worker processes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from langformer.verification.config import VerificationSettings


def _as_path_tuple(paths: Iterable[Union[str, Path]]) -> Tuple[Path, ...]:
    return tuple(Path(value).expanduser() for value in paths)


@dataclass(frozen=True)
class WorkerUnitSpec:
    """Description of the unit being transpiled."""

    id: str
    language: str
    kind: str = "module"
    source_code: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerUnitSpec":
        return cls(
            id=str(data.get("id", "")),
            language=str(data.get("language", "")),
            kind=str(data.get("kind", "module")),
            source_code=str(data.get("source_code", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "kind": self.kind,
            "source_code": self.source_code,
        }


@dataclass(frozen=True)
class WorkerContextSpec:
    """Subset of IntegrationContext relevant to workers."""

    target_language: str
    layout: Dict[str, Any] = field(default_factory=dict)
    build: Dict[str, Any] = field(default_factory=dict)
    api_mappings: Dict[str, Any] = field(default_factory=dict)
    feature_spec: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerContextSpec":
        return cls(
            target_language=str(data.get("target_language", "")),
            layout=dict(data.get("layout", {})),
            build=dict(data.get("build", {})),
            api_mappings=dict(data.get("api_mappings", {})),
            feature_spec=dict(data.get("feature_spec", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_language": self.target_language,
            "layout": self.layout,
            "build": self.build,
            "api_mappings": self.api_mappings,
            "feature_spec": self.feature_spec,
        }


@dataclass(frozen=True)
class WorkerAgentSettings:
    """DefaultTranspilerAgent-specific knobs for worker runs."""

    max_retries: int = 3
    temperature_range: Tuple[float, float] = (0.2, 0.6)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerAgentSettings":
        temps_value = data.get("temperature_range")
        temps_tuple: Tuple[float, ...]
        if isinstance(temps_value, Iterable) and not isinstance(
            temps_value, (str, bytes)
        ):
            temps_tuple = tuple(float(t) for t in temps_value)
        else:
            temps_tuple = (0.2, 0.6)
        temps_list = list(temps_tuple)
        if not temps_list:
            temps_list = [0.2, 0.6]
        return cls(
            max_retries=int(data.get("max_retries", 3)),
            temperature_range=(
                temps_list[0],
                temps_list[1] if len(temps_list) > 1 else temps_list[0],
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "temperature_range": self.temperature_range,
        }


@dataclass(frozen=True)
class WorkerEventsSettings:
    """Streaming/event configuration for worker processes."""

    enabled: bool = False
    base_dir: Optional[Path] = None
    model: str = "gpt-4o-mini"
    timeout_s: int = 120
    store_responses: bool = False
    mode: str = "file"
    variant_label: Optional[str] = None

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]]
    ) -> Optional["WorkerEventsSettings"]:
        if not data:
            return None
        return cls(
            enabled=bool(data.get("enabled")),
            base_dir=Path(data["base_dir"]).expanduser()
            if data.get("base_dir")
            else None,
            model=str(data.get("model", "gpt-4o-mini")),
            timeout_s=int(data.get("timeout_s", 120)),
            store_responses=bool(data.get("store_responses", False)),
            mode=str(data.get("mode", "file")),
            variant_label=data.get("variant_label"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_dir": str(self.base_dir) if self.base_dir else None,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "store_responses": self.store_responses,
            "mode": self.mode,
            "variant_label": self.variant_label,
        }


@dataclass(frozen=True)
class WorkerPayload:
    """Full payload consumed by worker processes."""

    source_language: str
    target_language: str
    unit: WorkerUnitSpec
    context: WorkerContextSpec
    llm: Dict[str, Any]
    prompt_paths: Tuple[Path, ...] = field(default_factory=tuple)
    agent: WorkerAgentSettings = field(default_factory=WorkerAgentSettings)
    verification: VerificationSettings = field(
        default_factory=VerificationSettings
    )
    events: Optional[WorkerEventsSettings] = None
    variant_label: Optional[str] = None
    shared_digests_dir: Optional[Path] = None

    @classmethod
    def from_raw(
        cls, data: Union["WorkerPayload", Dict[str, Any]]
    ) -> "WorkerPayload":
        if isinstance(data, WorkerPayload):
            return data
        prompt_paths = data.get("prompt_paths") or []
        shared_dir = data.get("shared_digests_dir")
        return cls(
            source_language=str(data.get("source_language", "")),
            target_language=str(data.get("target_language", "")),
            unit=WorkerUnitSpec.from_dict(data.get("unit", {})),
            context=WorkerContextSpec.from_dict(data.get("context", {})),
            llm=dict(data.get("llm", {})),
            prompt_paths=_as_path_tuple(prompt_paths),
            agent=WorkerAgentSettings.from_dict(data.get("agent", {})),
            verification=VerificationSettings.from_dict(
                data.get("verification", {})
            ),
            events=WorkerEventsSettings.from_dict(data.get("events")),
            variant_label=data.get("variant_label"),
            shared_digests_dir=Path(shared_dir).expanduser()
            if shared_dir
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "unit": self.unit.to_dict(),
            "context": self.context.to_dict(),
            "llm": self.llm,
            "prompt_paths": [str(path) for path in self.prompt_paths],
            "agent": self.agent.to_dict(),
            "verification": self.verification.to_dict(),
            "events": self.events.to_dict() if self.events else None,
            "variant_label": self.variant_label,
            "shared_digests_dir": (
                str(self.shared_digests_dir)
                if self.shared_digests_dir
                else None
            ),
        }
