"""LLM Transpiler agent that performs iterative prompt-driven refinement."""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from langformer.agents.base import LLMConfig
from langformer.exceptions import TranspilationAttemptError
from langformer.languages.base import LanguagePlugin
from langformer.prompting.backends.base import PromptRenderer
from langformer.prompting.backends.jinja_backend import JinjaPromptRenderer
from langformer.prompting.fills import PromptFillContext, prompt_fills
from langformer.prompting.types import PromptTaskResult, PromptTaskSpec
from langformer.runtime.parallel import ParallelExplorer
from langformer.types import (
    CandidatePatchSet,
    IntegrationContext,
    TranspileUnit,
)

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from langformer.agents.verifier import VerifierAgent

EventAdapterFactory = Callable[
    [TranspileUnit, IntegrationContext, int, Optional[str]], Any
]
DedupHandler = Callable[[str, int, Optional[str]], Optional[Dict[str, Any]]]


@runtime_checkable
class TranspilerAgent(Protocol):
    """Protocol for agents that turn units into candidate patch sets."""

    def transpile(
        self,
        unit: TranspileUnit,
        ctx: IntegrationContext,
        verifier: "VerifierAgent | None" = None,
        *,
        event_factory: Optional[EventAdapterFactory] = None,
        variant_label: Optional[str] = None,
        dedup_handler: Optional[DedupHandler] = None,
        cancel_event: Optional[Any] = None,
    ) -> CandidatePatchSet:
        """Generate code for a unit."""
        ...


class LLMTranspilerAgent:
    """Transpiler that uses prompt templates, providers, and LLM iterations."""

    def __init__(
        self,
        source_plugin: LanguagePlugin,
        target_plugin: LanguagePlugin,
        *,
        llm_config: LLMConfig,
        max_retries: int = 3,
        parallel_workers: int = 1,
        temperature_range: tuple[float, float] = (0.2, 0.6),
        event_adapter_factory: Optional[EventAdapterFactory] = None,
        prompt_renderer: PromptRenderer | None = None,
        prompt_template_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._source_plugin = source_plugin
        self._target_plugin = target_plugin
        self._llm_config = llm_config
        self._provider = llm_config.provider
        self._prompt_manager = llm_config.prompt_manager
        self._artifact_manager = llm_config.artifact_manager
        self._max_retries = max(1, max_retries)
        self._parallel_workers = max(1, parallel_workers)
        self._temperature_range = temperature_range
        provider_name = getattr(
            llm_config.provider, "__class__", type(llm_config.provider)
        ).__name__
        if provider_name == "ProviderAdapter":
            inner = getattr(llm_config.provider, "_provider", None)
            model_name = getattr(llm_config.provider, "_model_name", None)
            inner_name = (
                getattr(inner, "__class__", type(inner)).__name__
                if inner
                else "provider"
            )
            provider_name = (
                f"{inner_name} (model={model_name})"
                if model_name
                else inner_name
            )
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            "LLMTranspilerAgent using provider %s", provider_name
        )
        if provider_name.lower().startswith("echoprovider"):
            self._logger.warning(
                "LLM provider is EchoProvider; outputs will mirror prompts "
                "until a real provider is configured."
            )
        self._explorer = ParallelExplorer()
        self._event_factory = event_adapter_factory
        default_template_map = {
            "transpile_initial": "transpile.j2",
            "transpile_refine": "refine.j2",
        }
        if prompt_template_map:
            default_template_map.update(prompt_template_map)
        self._prompt_renderer = prompt_renderer or JinjaPromptRenderer(
            self._prompt_manager,
            template_map=default_template_map,
        )

    def transpile(
        self,
        unit: TranspileUnit,
        ctx: IntegrationContext,
        verifier: "VerifierAgent | None" = None,
        *,
        event_factory: Optional[EventAdapterFactory] = None,
        variant_label: Optional[str] = None,
        dedup_handler: Optional[DedupHandler] = None,
        cancel_event: Optional[Any] = None,
    ) -> CandidatePatchSet:
        """Iteratively prompt until verification passes (parallel optional)."""

        factory = event_factory or self._event_factory
        target_path = ctx.layout.target_path(unit.id)

        if verifier is None or self._parallel_workers == 1:
            return self._sequential_attempts(
                unit,
                ctx,
                target_path,
                verifier,
                event_factory=factory,
                variant_label=variant_label,
                dedup_handler=dedup_handler,
                cancel_event=cancel_event,
            )

        attempt_funcs = [
            lambda idx=idx: self._sequential_attempts(
                unit,
                ctx,
                target_path,
                verifier,
                temp=self._pick_temperature(idx),
                event_factory=factory,
                variant_label=(variant_label or "parallel") + f"_{idx:02d}",
                dedup_handler=dedup_handler,
                cancel_event=cancel_event,
            )
            for idx in range(self._parallel_workers)
        ]

        candidate = self._explorer.explore(attempt_funcs)
        if candidate is None:
            raise TranspilationAttemptError(
                f"All parallel attempts failed for unit {unit.id}"
            )
        return candidate

    def verify(self, unit, candidate, ctx):  # pragma: no cover - legacy hook
        raise NotImplementedError("Verifier is handled by VerifierAgent")

    def _sequential_attempts(
        self,
        unit: TranspileUnit,
        ctx: IntegrationContext,
        target_path: Path,
        verifier: "VerifierAgent | None",
        *,
        temp: Optional[float] = None,
        event_factory: Optional[EventAdapterFactory] = None,
        variant_label: Optional[str] = None,
        dedup_handler: Optional[DedupHandler] = None,
        cancel_event: Optional[Any] = None,
    ) -> CandidatePatchSet:
        feedback: Optional[str] = None
        previous_code: Optional[str] = None
        for attempt in range(1, self._max_retries + 1):
            if (
                cancel_event is not None
                and getattr(cancel_event, "is_set", lambda: False)()
            ):
                raise TranspilationAttemptError(
                    "Transpiler canceled by orchestrator"
                )
            task_spec = self._build_prompt_spec(
                unit, ctx, feedback, attempt, previous_code
            )
            rendered_prompt = self._prompt_renderer.render(task_spec)
            prompt = rendered_prompt.message.content
            stream_details: Optional[dict[str, Any]] = None
            adapter_instance: Optional[Any] = None
            if event_factory is not None:
                try:
                    adapter_instance = event_factory(
                        unit, ctx, attempt, variant_label
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self._logger.warning(
                        "Failed to initialize event adapter: %s", exc
                    )
            if adapter_instance is not None:
                try:
                    stream_details = adapter_instance.stream(
                        system_prompt=(
                            "You are a careful transpilation assistant."
                        ),
                        user_prompt=prompt,
                        extras={
                            "metadata": {
                                "unit": unit.id,
                                "attempt": attempt,
                                "variant": variant_label or "main",
                            }
                        },
                    )
                except Exception as exc:  # pragma: no cover
                    self._logger.warning("Event streaming failed: %s", exc)
                    stream_details = {"error": str(exc)}
            task_result: PromptTaskResult | None = None
            if stream_details is not None:
                task_result = PromptTaskResult(
                    output_type="code",
                    output=stream_details.get("output_text", "") or "",
                    metadata={
                        "source": "event_stream",
                        "response_id": stream_details.get("response_id"),
                    },
                )
            if task_result is None or not task_result.output:
                temperature_value = temp or self._pick_temperature(attempt)
                try:
                    generated = self._provider.generate(
                        prompt,
                        metadata={"source_code": unit.source_code or ""},
                        temperature=temperature_value,
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - surface provider failures
                    provider_name = getattr(
                        getattr(self._provider, "__class__", None),
                        "__name__",
                        "provider",
                    )
                    raise TranspilationAttemptError(
                        f"LLM provider '{provider_name}' failed: {exc}"
                    ) from exc
                task_result = PromptTaskResult(
                    output_type="code",
                    output=generated or "",
                    metadata={
                        "source": "provider",
                        "temperature": temperature_value,
                    },
                )
            assert task_result is not None
            code = task_result.output or ""
            notes: dict[str, Any] = {
                "attempt": attempt,
                "prompt_task": {
                    "kind": task_spec.kind,
                    "template": rendered_prompt.template,
                },
                "prompt_result": {
                    "output_type": task_result.output_type,
                    "source": task_result.metadata.get("source"),
                },
            }
            if variant_label:
                notes["variant"] = variant_label
            if stream_details is not None:
                notes["stream"] = {
                    "response_id": stream_details.get("response_id"),
                    "error": stream_details.get("error"),
                }
            dedup_info = None
            if dedup_handler is not None:
                try:
                    dedup_info = dedup_handler(code, attempt, variant_label)
                except Exception as exc:  # pragma: no cover - defensive
                    dedup_info = {"status": "error", "error": str(exc)}
                if dedup_info:
                    notes["dedup"] = dedup_info
                    status = dedup_info.get("status")
                    if status == "duplicate_same_worker":
                        feedback = json.dumps(
                            {
                                "reason": "duplicate_same_worker",
                                "dedup": dedup_info,
                            },
                        )
                        continue
                    if status == "duplicate_cross_worker":
                        raise TranspilationAttemptError(
                            "Duplicate candidate seen by another worker"
                        )

            candidate = CandidatePatchSet(
                files={target_path: code}, notes=notes
            )
            previous_code = code
            if verifier is None:
                return candidate

            result = verifier.verify(unit, candidate, ctx)
            verification_note = result.feedback.to_dict()
            candidate.notes["verification"] = verification_note
            if result.passed:
                return candidate
            feedback = json.dumps(verification_note)

        raise TranspilationAttemptError(
            f"Failed to transpile unit {unit.id} within retry budget"
        )

    def _build_prompt_spec(
        self,
        unit: TranspileUnit,
        ctx: IntegrationContext,
        feedback: Optional[str],
        attempt: int,
        previous_code: Optional[str],
    ) -> PromptTaskSpec:
        kind = (
            "transpile_initial" if feedback is None else "transpile_refine"
        )
        previous_snapshot = previous_code or unit.source_code or ""
        fill_context = PromptFillContext(
            unit=unit,
            integration_context=ctx,
            attempt=attempt,
            feedback=feedback,
            previous_code=previous_snapshot,
            source_language=self._source_plugin.name,
            target_language=self._target_plugin.name,
            source_plugin=self._source_plugin,
            target_plugin=self._target_plugin,
        )
        payload = prompt_fills.build_payload(fill_context)
        base_context = {
            "source_language": self._source_plugin.name,
            "target_language": self._target_plugin.name,
            "unit_kind": unit.kind,
            "source_code": unit.source_code or "",
            "feedback": feedback or "",
            "attempt": attempt,
            "previous_code": previous_snapshot,
        }
        base_context.update(payload)
        return PromptTaskSpec(
            kind=kind,
            task_id=str(unit.id),
            metadata=base_context,
        )

    def _pick_temperature(self, idx: int) -> float:
        low, high = self._temperature_range
        if self._max_retries <= 1:
            return low
        span = high - low
        fraction = min(1.0, idx / max(1, self._max_retries - 1))
        return low + span * fraction
