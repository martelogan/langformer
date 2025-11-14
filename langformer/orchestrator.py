"""Transpilation orchestrator that wires the scaffolded agents together."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import tempfile

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from dotenv import load_dotenv

from langformer.agents.analyzer import DefaultAnalyzerAgent
from langformer.agents.base import LLMConfig
from langformer.agents.transpiler import LLMTranspilerAgent, TranspilerAgent
from langformer.agents.verifier import DefaultVerificationAgent, VerifierAgent
from langformer.artifacts import ArtifactManager
from langformer.configuration import (
    AgentsSettings,
    build_transpilation_settings,
)
from langformer.constants import (
    ARTIFACT_STAGE_ANALYZER,
    ARTIFACT_STAGE_VERIFIER,
)
from langformer.exceptions import TranspilationAttemptError
from langformer.languages import LANGUAGE_PLUGINS, LanguagePlugin
from langformer.llm.providers import LLMProvider, load_provider
from langformer.logging import EventAdapter, StreamDispatcher
from langformer.orchestration.context_builder import ContextBuilder
from langformer.orchestration.target_integrator import TargetIntegrator
from langformer.prompts.manager import PromptManager
from langformer.runtime import RunSession
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    LayoutPlan,
    TranspileUnit,
    VerifyResult,
)
from langformer.verification.base import VerificationStrategy
from langformer.verification.config import VerificationSettings
from langformer.verification.factory import build_verification_strategy
from langformer.worker.manager import WorkerManager
from langformer.worker.payload import (
    WorkerAgentSettings,
    WorkerContextSpec,
    WorkerEventsSettings,
    WorkerPayload,
    WorkerUnitSpec,
)
from langformer.worker.transpile_worker import run_worker

DEFAULT_CONFIG_PATH = Path("configs/default_config.yaml")
DEFAULT_PROMPT_DIR = Path("langformer/prompts/templates")
_DOTENV_LOADED = False
LOGGER = logging.getLogger(__name__)


def _merge_layout_dicts(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _merge_layout_dicts(
                result.get(key, {}), value
            )
        else:
            result[key] = deepcopy(value)
    return result


class TranspilationConfigError(RuntimeError):
    """Raised when configuration is invalid."""


class TranspilationOrchestrator:
    """High level controller for the transpilation workflow."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        *,
        config: Optional[Dict] = None,
        source_plugin: Optional[LanguagePlugin] = None,
        target_plugin: Optional[LanguagePlugin] = None,
        verification_strategy: Optional[VerificationStrategy] = None,
    ) -> None:
        self.config_path = config_path
        self._config_root = (
            Path(config_path).resolve().parent
            if config_path is not None
            else Path.cwd()
        )
        self._ensure_dotenv()
        self.config = config or self._load_config(
            config_path or DEFAULT_CONFIG_PATH
        )
        self.settings = build_transpilation_settings(
            self.config, config_root=self._config_root
        )
        self._artifact_manager = ArtifactManager(self.settings.artifacts)
        self._import_plugin_modules(self.settings.plugins)
        self.source_plugin = source_plugin or self._instantiate_language(
            self.settings.source_language
        )
        self.target_plugin = target_plugin or self._instantiate_language(
            self.settings.target_language
        )
        self._llm_cfg = self.settings.llm.raw
        self._stream_settings = self.settings.llm.streaming
        self._stream_enabled = self._stream_settings.enabled
        self._stream_root = self._stream_settings.log_root
        self._stream_mode = self._stream_settings.mode
        self._worker_stream_mode = self._stream_settings.worker_mode
        self._llm_model_name = self._llm_cfg.get("model") or os.getenv(
            "OPENAI_MODEL", "gpt-4o-mini"
        )
        analysis_cfg = {"analysis": self.settings.analysis}
        components_cfg = self.settings.components
        self.context_builder = self._init_context_builder(
            components_cfg.get("context_builder")
        )
        agents_cfg = self.settings.agents
        temp_range = agents_cfg.temperature_range
        extra_prompt_dirs = list(agents_cfg.prompt_overrides)
        prompt_manager = PromptManager(
            DEFAULT_PROMPT_DIR, extra_dirs=extra_prompt_dirs
        )
        self._prompt_paths = prompt_manager.search_paths
        self._agent_cfg = {
            "max_retries": agents_cfg.max_retries,
            "temperature_range": temp_range,
            "parallel_workers": agents_cfg.parallel_workers,
        }
        self._worker_settings = agents_cfg.worker_manager
        runtime_cfg = self.settings.runtime
        self._runtime_enabled = runtime_cfg.enabled
        self._run_root = runtime_cfg.run_root
        self._run_root.mkdir(parents=True, exist_ok=True)
        self._resume_run_id = runtime_cfg.run_id
        self._resume_mode = runtime_cfg.resume or bool(self._resume_run_id)
        self._verification_settings = self.settings.verification
        provider: LLMProvider = load_provider(self._llm_cfg or {})

        self._llm_config = LLMConfig(
            provider=provider,
            prompt_manager=prompt_manager,
            prompt_paths=self._prompt_paths,
            artifact_manager=self._artifact_manager,
        )

        self.source_analyzer = self._init_source_analyzer(
            components_cfg.get("source_analyzer"),
            analysis_cfg,
            self._llm_config,
        )
        self.transpiler_agent = self._init_transpiler_agent(
            agents_cfg,
            components_cfg.get("transpiler_agent"),
            self._llm_config,
            temp_range,
        )
        verification_cfg = self._verification_settings
        strategy = verification_strategy or self._instantiate_verification(
            verification_cfg
        )
        self.verification_agent = self._init_verification_agent(
            components_cfg.get("verification_agent"),
            strategy,
            self._llm_config,
        )
        self.integration_agent = self._init_target_integrator(
            components_cfg.get("target_integrator")
        )

    def transpile_file(
        self, source_path: Path, target_path: Path, *, verify: bool = True
    ) -> Path:
        """Transpile a single source file into the target path."""
        source_path = Path(source_path)
        target_path = Path(target_path)
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source file {source_path} does not exist"
            )

        source_code = source_path.read_text()
        unit_id = source_path.stem
        candidate = self._run_pipeline(source_code, unit_id, verify=verify)
        self.integration_agent.integrate(candidate, destination=target_path)
        return target_path

    def transpile_directory(
        self, source_dir: Path, target_dir: Path, *, verify: bool = True
    ) -> None:
        """Transpile every file in a directory (non-recursive)."""
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for path in source_dir.iterdir():
            if path.is_file():
                destination = target_dir / path.name
                self.transpile_file(path, destination, verify=verify)

    def transpile_code(
        self, source_code: str, unit_id: str = "unit"
    ) -> CandidatePatchSet:
        """Transpile in-memory code and return the candidate patch set."""
        return self._run_pipeline(source_code, unit_id, verify=False)

    def _run_pipeline(
        self, source_code: str, unit_id: str, *, verify: bool
    ) -> CandidatePatchSet:
        layout_plan = self._current_layout_plan()
        self._artifact_manager.reset_all()
        context: IntegrationContext = self.context_builder.build(
            self.settings.integration,
            layout_plan,
            artifacts=self._artifact_manager,
        )
        units: List[TranspileUnit] = self.source_analyzer.analyze(
            source_code, unit_id=unit_id
        )
        session: Optional[RunSession] = None
        if self._runtime_enabled:
            if self._resume_mode and self._resume_run_id:
                session = RunSession(
                    self._run_root, run_id=self._resume_run_id, resume=True
                )
            else:
                session = RunSession(self._run_root)
            session.write_metadata(
                "meta",
                {
                    "run_id": session.run_id,
                    "unit_id": unit_id,
                    "num_units": len(units),
                    "target_language": self.target_plugin.name,
                },
            )
        event_factory = self._build_event_factory(session)

        resume_candidates: Dict[str, CandidatePatchSet] = {}
        if session is not None and session.resume:
            for entry in session.list_units(status="success"):
                unit_identifier = entry.get("unit_id")
                if not unit_identifier:
                    continue
                files = session.load_files(unit_identifier)
                if not files:
                    continue
                candidate = CandidatePatchSet(
                    files={
                        Path(path): contents
                        for path, contents in files.items()
                    },
                    notes=entry.get("result", {}).get("notes", {}),
                )
                resume_candidates[unit_identifier] = candidate

        use_workers = (
            self._worker_settings.enabled and verify and context.oracle is None
        )
        candidates: List[CandidatePatchSet] = []
        LOGGER.info(
            "Preparing %d transpilation unit(s) for '%s' (verify=%s)",
            len(units),
            unit_id,
            verify,
        )
        for unit in units:
            self._artifact_manager.reset(
                unit.id,
                preserve=[ARTIFACT_STAGE_ANALYZER],
            )
            unit_metadata = {
                "language": unit.language,
                "kind": unit.kind,
                "strategy": "worker" if use_workers else "inline",
            }
            self._log_unit_started(session, unit.id, unit_metadata)
            try:
                if (
                    session is not None
                    and session.resume
                    and unit.id in resume_candidates
                ):
                    candidate = resume_candidates[unit.id]
                    self._log_unit_skipped(
                        session, unit.id, "resume_cached_result"
                    )
                elif use_workers:
                    LOGGER.info("Dispatching unit %s to worker pool", unit.id)
                    candidate = self._transpile_with_workers(
                        unit, context, session=session
                    )
                else:
                    LOGGER.info(
                        "Running inline transpilation for unit %s", unit.id
                    )
                    candidate = self.transpiler_agent.transpile(
                        unit,
                        context,
                        verifier=self.verification_agent if verify else None,
                        event_factory=event_factory,
                        variant_label="orchestrator",
                    )
            except TranspilationAttemptError as exc:
                self._log_unit_failed(session, unit.id, {"error": str(exc)})
                raise TranspilationAttemptError(str(exc)) from exc
            if verify and "verification" not in candidate.notes:
                verification_result = self.verification_agent.verify(
                    unit, candidate, context
                )
                if not verification_result.passed:
                    self._record_verification_artifacts(
                        unit.id, verification_result, candidate
                    )
                    self._record_artifacts(unit.id, candidate, session)
                    self._log_unit_failed(
                        session,
                        unit.id,
                        {"verification": verification_result.details},
                    )
                    raise TranspilationAttemptError(
                        f"Verification failed: {verification_result.details}"
                    )
                LOGGER.info("Verification succeeded for unit %s", unit.id)
                self._record_verification_artifacts(
                    unit.id, verification_result, candidate
                )
            self._record_artifacts(unit.id, candidate, session)
            candidates.append(candidate)
            if session is not None:
                manifest = session.persist_files(unit.id, candidate.files)
                self._log_unit_succeeded(
                    session,
                    unit.id,
                    {
                        "notes": candidate.metadata.to_dict(),
                        "files": [str(p) for p in candidate.files.keys()],
                        "manifest": manifest,
                    },
                )

        if not candidates:
            raise TranspilationConfigError(
                "Analyzer did not return any units to transpile"
            )
        if len(candidates) == 1:
            LOGGER.info("Transpilation completed for unit %s", unit_id)
            return candidates[0]
        output_path = context.layout.target_path(unit_id)
        result = self.integration_agent.combine_candidates(
            candidates, output_path
        )
        if session is not None:
            session.write_metadata(
                "result", {"files": [str(p) for p in result.files.keys()]}
            )
            session.write_summary(
                {"result_files": [str(p) for p in result.files.keys()]}
            )
        LOGGER.info("Transpilation completed for unit %s", unit_id)
        return result

    def _log_unit_started(
        self,
        session: Optional[RunSession],
        unit_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        if session is not None:
            session.mark_unit_started(unit_id, metadata)
            session.log_event("unit_started", {"unit": unit_id, **metadata})
        LOGGER.info(
            "Unit %s started (strategy=%s)", unit_id, metadata.get("strategy")
        )

    def _log_unit_skipped(
        self, session: Optional[RunSession], unit_id: str, reason: str
    ) -> None:
        if session is not None:
            session.log_event(
                "unit_skipped", {"unit": unit_id, "reason": reason}
            )
        LOGGER.info("Unit %s skipped (%s)", unit_id, reason)

    def _log_unit_failed(
        self,
        session: Optional[RunSession],
        unit_id: str,
        details: Dict[str, Any],
    ) -> None:
        if session is not None:
            session.mark_unit_completed(unit_id, "failed", details)
            session.log_event("unit_failed", {"unit": unit_id, **details})
        LOGGER.error("Unit %s failed: %s", unit_id, details)

    def _log_unit_succeeded(
        self,
        session: Optional[RunSession],
        unit_id: str,
        details: Dict[str, Any],
    ) -> None:
        if session is not None:
            session.mark_unit_completed(unit_id, "success", details)
            session.log_event("unit_succeeded", {"unit": unit_id, **details})
        LOGGER.info("Unit %s succeeded", unit_id)

    def _record_artifacts(
        self,
        unit_id: str,
        candidate: CandidatePatchSet,
        session: Optional[RunSession],
    ) -> None:
        manifest = self._artifact_manager.manifest_for(unit_id)
        if not manifest:
            return
        artifacts_note = candidate.notes.setdefault("artifacts", {})
        for stage, entries in manifest.items():
            stage_entries = artifacts_note.setdefault(stage, [])
            stage_entries.extend(entries)
        if session is not None:
            session.write_metadata(
                f"{unit_id}_artifacts",
                {"artifacts": manifest},
            )

    def _record_verification_artifacts(
        self,
        unit_id: str,
        result: VerifyResult,
        candidate: CandidatePatchSet,
    ) -> None:
        tests = result.feedback.tests
        if not tests:
            return
        stage_dir = self._artifact_manager.stage_dir(
            ARTIFACT_STAGE_VERIFIER, unit_id
        )
        for rel_path, contents in tests.items():
            rel = Path(rel_path)
            file_path = stage_dir / rel
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents, encoding="utf-8")
            candidate.add_test(rel, contents)
            self._artifact_manager.register(
                ARTIFACT_STAGE_VERIFIER,
                unit_id,
                file_path,
                metadata={"kind": "verification_test"},
            )

    def _transpile_with_workers(
        self,
        unit: TranspileUnit,
        context: IntegrationContext,
        *,
        session: Optional[RunSession],
    ) -> CandidatePatchSet:
        worker_cfg = self._worker_settings
        worker_count = worker_cfg.active_workers or self._agent_cfg.get(
            "parallel_workers",
            2,
        )
        variants = worker_cfg.active_variants
        tasks: List[WorkerPayload] = []
        if session is not None:
            shared_digests_dir = session.dirs.digests
        else:
            shared_digests_dir = self._create_shared_digests_dir()
        worker_agent = WorkerAgentSettings(
            max_retries=self._agent_cfg["max_retries"],
            temperature_range=self._agent_cfg["temperature_range"],
        )
        for variant in range(variants):
            variant_label = f"variant_{variant:02d}"
            events_settings = self._build_worker_event_settings(
                session, unit, variant_label
            )
            payload = WorkerPayload(
                source_language=unit.language,
                target_language=context.target_language,
                unit=WorkerUnitSpec(
                    id=unit.id,
                    language=unit.language,
                    kind=unit.kind,
                    source_code=unit.source_code or "",
                ),
                context=WorkerContextSpec(
                    target_language=context.target_language,
                    layout=context.layout.as_dict(),
                    build=context.build,
                    api_mappings=context.api_mappings,
                    feature_spec=context.feature_spec,
                ),
                llm=self.config.get("transpilation", {}).get(
                    "llm",
                    {"provider": "echo"},
                ),
                prompt_paths=self._prompt_paths,
                agent=worker_agent,
                verification=self._verification_settings,
                events=events_settings,
                variant_label=variant_label,
                shared_digests_dir=shared_digests_dir,
            )
            tasks.append(payload)
        log_dir_cfg = worker_cfg.log_dir
        if log_dir_cfg:
            log_dir = Path(log_dir_cfg)
        elif session is not None:
            log_dir = session.dirs.workers / unit.id
        else:
            log_dir = None
        manager = WorkerManager(
            num_workers=worker_count, worker_fn=run_worker, log_dir=log_dir
        )
        result = manager.run(tasks)
        if not result:
            raise TranspilationAttemptError(
                "Worker manager failed to produce a candidate"
            )
        files = {
            Path(path): contents for path, contents in result["files"].items()
        }
        candidate = CandidatePatchSet(
            files=files, notes=result.get("notes", {})
        )
        if session is not None:
            session.write_metadata(
                f"{unit.id}_worker",
                {
                    "files": [str(p) for p in candidate.files],
                    "notes": candidate.notes,
                },
            )
        return candidate

    def _create_shared_digests_dir(self) -> Path:
        prefix = "digests_"
        base_dir: Optional[Path] = None
        if self._run_root:
            try:
                base_dir = Path(self._run_root)
                base_dir.mkdir(parents=True, exist_ok=True)
                return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_dir)))
            except (
                PermissionError,
                FileNotFoundError,
            ):  # pragma: no cover - fallback
                pass
        return Path(tempfile.mkdtemp(prefix=prefix))

    def _load_config(self, path: Path) -> Dict:
        if not Path(path).exists():
            raise TranspilationConfigError(f"Config file {path} not found")
        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _instantiate_language(self, name: Optional[str]) -> LanguagePlugin:
        if not name:
            raise TranspilationConfigError("Language name missing in config")
        try:
            return LANGUAGE_PLUGINS[name.lower()]()
        except KeyError as exc:  # pragma: no cover - future languages
            raise TranspilationConfigError(
                f"Unknown language plugin '{name}'"
            ) from exc

    def _instantiate_verification(
        self,
        config: VerificationSettings,
    ) -> VerificationStrategy:
        try:
            return build_verification_strategy(
                config,
                source_plugin=self.source_plugin,
                target_plugin=self.target_plugin,
            )
        except ValueError as exc:
            raise TranspilationConfigError(str(exc)) from exc

    def get_config(self) -> Dict:
        """Return the loaded configuration."""
        return self.config

    def _import_plugin_modules(self, plugin_cfg: Dict[str, Any]) -> None:
        paths = list(plugin_cfg.get("paths") or [])
        modules = plugin_cfg.get("modules") or []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = (self._config_root / path).resolve()
            if not path.exists():
                raise TranspilationConfigError(
                    f"Plugin path '{path}' does not exist"
                )
            for candidate in filter(None, {path, path.parent}):
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
        try:
            current_dir = Path(__file__).resolve().parents[1]
        except Exception:
            current_dir = None
        if current_dir is not None and str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        for dotted in modules:
            try:
                importlib.import_module(dotted)
            except Exception as exc:  # pragma: no cover
                raise TranspilationConfigError(
                    f"Failed to import plugin module '{dotted}': {exc}"
                ) from exc

    def _build_event_factory(
        self, session: Optional[RunSession]
    ) -> Optional[Callable]:
        if not self._stream_enabled:
            return None
        store_responses = self._stream_settings.store_responses
        timeout_s = self._stream_settings.timeout_s
        base_dir = self._resolve_stream_base(session, scope="orchestrator")

        def factory(
            unit: TranspileUnit,
            ctx: IntegrationContext,
            attempt: int,
            variant_label: Optional[str],
        ) -> EventAdapter:
            label = variant_label or "main"
            unit_dir = base_dir / unit.id / label
            unit_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = unit_dir / f"attempt_{attempt:02d}.jsonl"
            console_path = unit_dir / "console.log"
            dispatcher = StreamDispatcher(console_path, mode=self._stream_mode)
            return EventAdapter(
                model=self._llm_model_name,
                store_responses=store_responses,
                timeout_s=timeout_s,
                jsonl_path=jsonl_path,
                on_delta=lambda chunk,
                d=dispatcher,
                u=unit.id,
                lbl=label,
                att=attempt: d.emit(
                    chunk,
                    prefix=f"{u}:{lbl}:{att:02d}",
                ),
            )

        return factory

    def _ensure_dotenv(self) -> None:
        global _DOTENV_LOADED
        if _DOTENV_LOADED:
            return
        try:  # pragma: no cover
            load_dotenv()
        except Exception:
            pass
        _DOTENV_LOADED = True

    def _build_worker_event_settings(
        self,
        session: Optional[RunSession],
        unit: TranspileUnit,
        variant_label: str,
    ) -> Optional[WorkerEventsSettings]:
        if not self._stream_enabled:
            return None
        base_dir = self._resolve_stream_base(session, scope="workers")
        variant_root = base_dir / unit.id / variant_label
        variant_root.mkdir(parents=True, exist_ok=True)
        return WorkerEventsSettings(
            enabled=True,
            base_dir=variant_root,
            model=self._llm_model_name,
            timeout_s=self._stream_settings.timeout_s,
            store_responses=self._stream_settings.store_responses,
            mode=self._worker_stream_mode,
            variant_label=variant_label,
        )

    def _current_layout_plan(self) -> LayoutPlan:
        runtime_layout = (
            self.config.get("transpilation", {}).get("layout")
        )
        merged = _merge_layout_dicts(
            self.settings.layout_raw,
            runtime_layout,
        )
        return LayoutPlan(merged)

    def _resolve_stream_base(
        self, session: Optional[RunSession], scope: str
    ) -> Path:
        if session is not None:
            root = (
                session.dirs.orchestrator
                if scope == "orchestrator"
                else session.dirs.workers
            )
            base = root / "streams"
        else:
            base = self._stream_root / scope
        base.mkdir(parents=True, exist_ok=True)
        return base

    # ------------------------------------------------------------------
    # Component initializers
    # ------------------------------------------------------------------

    def _init_source_analyzer(
        self,
        class_path: Optional[str],
        analysis_cfg: Dict[str, Any],
        llm_config: LLMConfig,
    ):
        if class_path:
            cls = self._import_symbol(class_path)
            return cls(self.source_plugin, llm_config, config=analysis_cfg)
        return DefaultAnalyzerAgent(
            self.source_plugin, llm_config, config=analysis_cfg
        )

    def _init_context_builder(
        self, class_path: Optional[str]
    ) -> ContextBuilder:
        if class_path:
            cls = self._import_symbol(class_path)
            return cls()
        return ContextBuilder()

    def _init_transpiler_agent(
        self,
        agents_cfg: AgentsSettings,
        class_path: Optional[str],
        llm_config: LLMConfig,
        temp_range,
    ) -> TranspilerAgent:
        ctor = LLMTranspilerAgent
        if class_path:
            ctor = self._import_symbol(class_path)
        return ctor(  # type: ignore[call-arg]
            self.source_plugin,
            self.target_plugin,
            llm_config=llm_config,
            max_retries=self._agent_cfg["max_retries"],
            parallel_workers=agents_cfg.parallel_workers,
            temperature_range=temp_range,
        )

    def _init_verification_agent(
        self,
        class_path: Optional[str],
        strategy: VerificationStrategy,
        llm_config: LLMConfig,
    ) -> VerifierAgent:
        ctor = DefaultVerificationAgent
        if class_path:
            ctor = self._import_symbol(class_path)
        return ctor(  # type: ignore[call-arg]
            strategy,
            source_plugin=self.source_plugin,
            target_plugin=self.target_plugin,
            llm_config=llm_config,
        )

    def _init_target_integrator(
        self, class_path: Optional[str]
    ) -> TargetIntegrator:
        if class_path:
            cls = self._import_symbol(class_path)
            return cls()
        return TargetIntegrator()

    def _import_symbol(self, dotted_path: str):
        try:
            module_path, attr = dotted_path.rsplit(".", 1)
        except ValueError as exc:  # pragma: no cover
            raise TranspilationConfigError(
                f"Invalid dotted path '{dotted_path}'"
            ) from exc
        module = importlib.import_module(module_path)
        try:
            return getattr(module, attr)
        except AttributeError as exc:  # pragma: no cover
            raise TranspilationConfigError(
                f"{dotted_path} is not importable"
            ) from exc
