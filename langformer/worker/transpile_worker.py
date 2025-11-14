# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Worker entry point for multi-variant LLM transpilation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from langformer.agents.base import LLMConfig
from langformer.agents.transpiler import LLMTranspilerAgent
from langformer.agents.verifier import DefaultVerificationAgent
from langformer.exceptions import TranspilationAttemptError
from langformer.languages import LANGUAGE_PLUGINS
from langformer.llm.providers import load_provider
from langformer.logging import EventAdapter, StreamDispatcher
from langformer.prompts.manager import PromptManager
from langformer.runtime.dedup import CodeDeduplicator
from langformer.types import IntegrationContext, LayoutPlan, TranspileUnit
from langformer.verification.factory import build_verification_strategy
from langformer.worker.payload import WorkerEventsSettings, WorkerPayload


def run_worker(
    worker_id: int,
    payload: Union[WorkerPayload, Dict[str, Any]],
) -> Dict[str, Any]:
    task = WorkerPayload.from_raw(payload)
    source_plugin = LANGUAGE_PLUGINS[task.source_language]()
    target_plugin = LANGUAGE_PLUGINS[task.target_language]()
    provider = load_provider(task.llm)
    prompt_manager = PromptManager(search_paths=task.prompt_paths or None)
    resolved_paths = prompt_manager.search_paths
    llm_config = LLMConfig(
        provider=provider,
        prompt_manager=prompt_manager,
        prompt_paths=resolved_paths,
    )
    agent = LLMTranspilerAgent(
        source_plugin,
        target_plugin,
        llm_config=llm_config,
        max_retries=task.agent.max_retries,
        parallel_workers=1,
        temperature_range=task.agent.temperature_range,
    )
    event_factory = _build_event_factory(worker_id, task.events)
    shared_dir = task.shared_digests_dir
    worker_label = task.variant_label or f"worker_{worker_id:02d}"
    dedup_handler = None
    if shared_dir:
        dedup = CodeDeduplicator(Path(shared_dir), worker_label)

        def _dedup_handler(
            code: str, attempt: int, variant_label: Optional[str]
        ) -> Dict[str, Any]:
            return dedup.register(code, attempt)

        dedup_handler = _dedup_handler
    unit = TranspileUnit(
        id=task.unit.id,
        language=task.unit.language,
        kind=task.unit.kind,
        source_code=task.unit.source_code,
    )
    context = IntegrationContext(
        target_language=task.context.target_language,
        runtime_adapter=None,
        contract=None,
        layout=LayoutPlan(task.context.layout),
        build=task.context.build,
        oracle=None,
        api_mappings=task.context.api_mappings,
        feature_spec=task.context.feature_spec,
    )
    strategy = build_verification_strategy(
        task.verification,
        source_plugin=source_plugin,
        target_plugin=target_plugin,
    )
    verifier = DefaultVerificationAgent(
        strategy,
        source_plugin=source_plugin,
        target_plugin=target_plugin,
        llm_config=llm_config,
    )
    try:
        candidate = agent.transpile(
            unit,
            context,
            verifier=verifier,
            event_factory=event_factory,
            variant_label=task.variant_label,
            dedup_handler=dedup_handler,
        )
    except TranspilationAttemptError as exc:
        return {
            "success": False,
            "error": str(exc),
            "notes": {"variant": task.variant_label},
        }
    files = {str(path): content for path, content in candidate.files.items()}
    return {
        "success": True,
        "files": files,
        "notes": candidate.metadata.to_dict(),
    }


def _build_event_factory(
    worker_id: int,
    cfg: Optional[WorkerEventsSettings],
) -> Optional[Any]:
    if not cfg or not cfg.enabled or not cfg.base_dir:
        return None
    base_dir = cfg.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    model = cfg.model
    timeout_s = cfg.timeout_s
    store_responses = cfg.store_responses
    mode = cfg.mode

    def factory(
        unit: TranspileUnit,
        ctx: IntegrationContext,
        attempt: int,
        variant_label: Optional[str],
    ) -> EventAdapter:
        label = variant_label or cfg.variant_label or f"worker_{worker_id}"
        variant_dir = base_dir / unit.id / label
        variant_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = variant_dir / f"attempt_{attempt:02d}.jsonl"
        dispatcher = StreamDispatcher(variant_dir / "console.log", mode=mode)
        return EventAdapter(
            model=model,
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
