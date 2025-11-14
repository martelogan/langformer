"""Typed helpers for parsing Langformer configuration dictionaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from langformer.constants import (
    ARTIFACT_STAGE_ANALYZER,
    ARTIFACT_STAGE_TRANSPILER,
    ARTIFACT_STAGE_VERIFIER,
)
from langformer.verification.config import (
    VerificationSettings,
    build_verification_settings,
)


def _ensure_path(
    value: Optional[str | Path],
    *,
    config_root: Path,
    default: Optional[Path] = None,
) -> Path:
    if value is None:
        if default is None:
            raise ValueError("Path value is required")
        return default
    path = Path(value)
    if not path.is_absolute():
        path = (config_root / path).resolve()
    return path


def _coerce_temperature_range(
    value: Optional[Iterable[float]],
) -> Tuple[float, float]:
    if not value:
        return (0.2, 0.6)
    data = list(value)
    if len(data) == 1:
        return (float(data[0]), float(data[0]))
    return (float(data[0]), float(data[1]))


def _extract_oracle_config(
    transp_cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    verification_cfg = transp_cfg.get("verification") or {}
    oracle_cfg = verification_cfg.get("oracle")
    if oracle_cfg:
        return deepcopy(oracle_cfg)
    legacy_oracle = transp_cfg.get("oracle")
    if legacy_oracle:
        return deepcopy(legacy_oracle)
    return None


@dataclass(frozen=True)
class StreamingSettings:
    enabled: bool = False
    log_root: Path = field(
        default_factory=lambda: Path(".transpile_streams").resolve()
    )
    mode: str = "file"
    worker_mode: str = "file"
    store_responses: bool = False
    timeout_s: int = 120


@dataclass(frozen=True)
class LLMSettings:
    raw: Dict[str, Any] = field(default_factory=dict)
    streaming: StreamingSettings = field(default_factory=StreamingSettings)


@dataclass(frozen=True)
class WorkerManagerSettings:
    enabled: bool = False
    workers: int = 0
    variants_per_unit: int = 0
    log_dir: Optional[str] = None

    @property
    def active_workers(self) -> int:
        return max(1, self.workers or 0)

    @property
    def active_variants(self) -> int:
        if self.variants_per_unit > 0:
            return self.variants_per_unit
        return self.active_workers


@dataclass(frozen=True)
class AgentsSettings:
    max_retries: int = 3
    parallel_workers: int = 1
    temperature_range: Tuple[float, float] = (0.2, 0.6)
    prompt_overrides: Tuple[Path, ...] = field(default_factory=tuple)
    worker_manager: WorkerManagerSettings = field(
        default_factory=WorkerManagerSettings
    )


@dataclass(frozen=True)
class RuntimeSettings:
    enabled: bool = False
    run_root: Path = field(
        default_factory=lambda: Path(".transpile_runs").resolve()
    )
    run_id: Optional[str] = None
    resume: bool = False


@dataclass(frozen=True)
class LayoutIO:
    path: Optional[str] = None
    kind: str = "file"
    filename: Optional[str] = None
    extension: Optional[str] = None


@dataclass(frozen=True)
class LayoutSettings:
    relative_to: str = "cwd"
    input: LayoutIO = field(default_factory=LayoutIO)
    output: LayoutIO = field(default_factory=LayoutIO)
    module_path: Optional[str] = None


@dataclass(frozen=True)
class ArtifactSettings:
    root: Path
    analyzer_dir: str = ARTIFACT_STAGE_ANALYZER
    transpiler_dir: str = ARTIFACT_STAGE_TRANSPILER
    verifier_dir: str = ARTIFACT_STAGE_VERIFIER


@dataclass(frozen=True)
class IntegrationSettings:
    target_language: str
    runtime_adapter: Optional[str] = None
    contract: Optional[Any] = None
    build: Dict[str, Any] = field(default_factory=dict)
    api_mappings: Dict[str, Any] = field(default_factory=dict)
    feature_spec: Dict[str, Any] = field(default_factory=dict)
    oracle: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class TranspilationSettings:
    source_language: str
    target_language: str
    integration: IntegrationSettings
    layout_raw: Dict[str, Any]
    layout: LayoutSettings
    analysis: Dict[str, Any]
    components: Dict[str, Any]
    plugins: Dict[str, Any]
    llm: LLMSettings
    agents: AgentsSettings
    runtime: RuntimeSettings
    artifacts: ArtifactSettings
    verification_raw: Dict[str, Any]
    verification: VerificationSettings


def build_transpilation_settings(
    config: Dict[str, Any], *, config_root: Path
) -> TranspilationSettings:
    transp_cfg = config.get("transpilation") or {}
    source_language = transp_cfg.get("source_language") or "python"
    target_language = transp_cfg.get("target_language") or "python"

    llm_cfg = dict(transp_cfg.get("llm") or {})
    streaming_cfg = dict(llm_cfg.get("streaming") or {})
    log_root = _ensure_path(
        streaming_cfg.get("log_root", ".transpile_streams"),
        config_root=config_root,
        default=Path(".transpile_streams").resolve(),
    )
    streaming = StreamingSettings(
        enabled=bool(streaming_cfg.get("enabled", False)),
        log_root=log_root,
        mode=str(streaming_cfg.get("mode", "file")),
        worker_mode=str(streaming_cfg.get("worker_mode", "file")),
        store_responses=bool(streaming_cfg.get("store_responses", False)),
        timeout_s=int(streaming_cfg.get("timeout_s", 120)),
    )
    llm_settings = LLMSettings(raw=llm_cfg, streaming=streaming)

    agents_cfg = transp_cfg.get("agents") or {}
    prompt_dir_value = agents_cfg.get("prompt_dir")
    prompt_overrides: Tuple[Path, ...] = tuple()
    if prompt_dir_value:
        dirs: list[str | Path]
        if isinstance(prompt_dir_value, (list, tuple)):
            dirs = list(prompt_dir_value)
        else:
            dirs = [prompt_dir_value]
        prompt_overrides = tuple(
            _ensure_path(item, config_root=config_root) for item in dirs
        )
    worker_cfg = dict(agents_cfg.get("worker_manager") or {})
    worker_settings = WorkerManagerSettings(
        enabled=bool(worker_cfg.get("enabled")),
        workers=int(worker_cfg.get("workers", 0)),
        variants_per_unit=int(worker_cfg.get("variants_per_unit", 0)),
        log_dir=worker_cfg.get("log_dir"),
    )

    agents_settings = AgentsSettings(
        max_retries=int(agents_cfg.get("max_retries", 3)),
        parallel_workers=int(agents_cfg.get("parallel_workers", 1)),
        temperature_range=_coerce_temperature_range(
            agents_cfg.get("temperature_range")
        ),
        prompt_overrides=prompt_overrides,
        worker_manager=worker_settings,
    )

    runtime_cfg = transp_cfg.get("runtime") or {}
    run_root = _ensure_path(
        runtime_cfg.get("run_root", ".transpile_runs"),
        config_root=config_root,
        default=Path(".transpile_runs").resolve(),
    )
    runtime_settings = RuntimeSettings(
        enabled=bool(runtime_cfg.get("enabled", False)),
        run_root=run_root,
        run_id=runtime_cfg.get("run_id"),
        resume=bool(runtime_cfg.get("resume", False)),
    )

    verification_cfg = transp_cfg.get("verification") or {}
    verification_settings = build_verification_settings(
        verification_cfg,
        config_root=config_root,
    )

    integration_settings = IntegrationSettings(
        target_language=str(target_language),
        runtime_adapter=transp_cfg.get("runtime_adapter"),
        contract=transp_cfg.get("contract"),
        build=deepcopy(transp_cfg.get("build") or {}),
        api_mappings=deepcopy(transp_cfg.get("api_mappings") or {}),
        feature_spec=deepcopy(transp_cfg.get("feature_spec") or {}),
        oracle=_extract_oracle_config(transp_cfg),
    )

    layout_cfg = deepcopy(transp_cfg.get("layout") or {})
    input_cfg = deepcopy(layout_cfg.get("input") or {})
    output_cfg = deepcopy(layout_cfg.get("output") or {})
    layout_settings = LayoutSettings(
        relative_to=str(layout_cfg.get("relative_to", "cwd")).lower(),
        input=LayoutIO(
            path=input_cfg.get("path"),
            kind=str(input_cfg.get("kind", "file")),
            filename=input_cfg.get("filename"),
            extension=input_cfg.get("extension"),
        ),
        output=LayoutIO(
            path=output_cfg.get("path"),
            kind=str(output_cfg.get("kind", "file")),
            filename=output_cfg.get("filename"),
            extension=output_cfg.get("extension"),
        ),
        module_path=layout_cfg.get("module_path"),
    )

    artifacts_cfg = transp_cfg.get("artifacts") or {}
    artifact_root_value = artifacts_cfg.get("root")
    if artifact_root_value:
        artifact_root = _ensure_path(
            artifact_root_value,
            config_root=config_root,
            default=runtime_settings.run_root / "artifacts",
        )
    else:
        artifact_root = (runtime_settings.run_root / "artifacts").resolve()
    artifact_settings = ArtifactSettings(
        root=artifact_root,
        analyzer_dir=str(
            artifacts_cfg.get("analyzer_dir", ARTIFACT_STAGE_ANALYZER)
        ),
        transpiler_dir=str(
            artifacts_cfg.get("transpiler_dir", ARTIFACT_STAGE_TRANSPILER)
        ),
        verifier_dir=str(
            artifacts_cfg.get("verifier_dir", ARTIFACT_STAGE_VERIFIER)
        ),
    )

    return TranspilationSettings(
        source_language=str(source_language),
        target_language=str(target_language),
        integration=integration_settings,
        layout_raw=deepcopy(layout_cfg),
        layout=layout_settings,
        analysis=deepcopy(transp_cfg.get("analysis") or {}),
        components=deepcopy(transp_cfg.get("components") or {}),
        plugins=deepcopy(transp_cfg.get("plugins") or {}),
        llm=llm_settings,
        agents=agents_settings,
        runtime=runtime_settings,
        artifacts=artifact_settings,
        verification_raw=deepcopy(verification_cfg),
        verification=verification_settings,
    )


def _layout_base_dir(
    layout: LayoutSettings,
    *,
    config_base: Path,
    run_root: Path,
) -> Path:
    option = layout.relative_to.lower()
    if option == "config":
        return config_base
    if option == "run_root":
        return run_root
    return Path.cwd()


def resolve_input_path(
    layout: LayoutSettings,
    *,
    config_base: Path,
    run_root: Path,
) -> Path:
    if not layout.input.path:
        raise ValueError("layout.input.path is required")
    base_dir = _layout_base_dir(
        layout, config_base=config_base, run_root=run_root
    )
    path = Path(layout.input.path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_output_path(
    layout: LayoutSettings,
    *,
    source: Path,
    config_base: Path,
    run_root: Path,
) -> Path:
    base_dir = _layout_base_dir(
        layout, config_base=config_base, run_root=run_root
    )
    output = layout.output
    kind = output.kind.lower()
    if kind == "directory":
        directory = Path(output.path).expanduser() if output.path else base_dir
        if not directory.is_absolute():
            directory = (base_dir / directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        filename = output.filename
        if not filename:
            extension = output.extension or ".out"
            if extension and not extension.startswith("."):
                extension = f".{extension}"
            filename = f"{source.stem}{extension}"
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    if output.path:
        path = Path(output.path).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if layout.module_path:
        path = Path(layout.module_path).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    fallback = (run_root / f"{source.name}.out").resolve()
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


__all__ = [
    "StreamingSettings",
    "LLMSettings",
    "AgentsSettings",
    "WorkerManagerSettings",
    "RuntimeSettings",
    "LayoutSettings",
    "LayoutIO",
    "ArtifactSettings",
    "IntegrationSettings",
    "TranspilationSettings",
    "build_transpilation_settings",
    "resolve_input_path",
    "resolve_output_path",
]
