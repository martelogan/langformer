"""CLI entrypoints for the Langformer framework."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from dotenv import load_dotenv

from langformer import TranspilationOrchestrator
from langformer.configuration import (
    build_transpilation_settings,
    resolve_input_path,
    resolve_output_path,
)
from langformer.orchestration.context_builder import ContextBuilder
from langformer.orchestrator import DEFAULT_CONFIG_PATH
from langformer.preprocessing.delegates import load_delegate
from langformer.preprocessing.planners import ExecutionPlan, load_planner
from langformer.types import CandidatePatchSet, LayoutPlan, TranspileUnit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transpile source code using the Langformer."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=str,
        help="Path to the source file to transpile.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        help=(
            "Destination file for the transpiled output "
            "(default: <source>.out)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help=(
            "Path to a YAML config. If omitted, uses "
            "configs/default_config.yaml."
        ),
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Directory where run artifacts (RunSession) are written.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        help=(
            "Override the LLM provider "
            "(e.g., openai, anthropic, relay, echo)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the LLM model name.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification when set.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List existing run directories under --run-root and exit.",
    )
    parser.add_argument(
        "--show-run",
        type=str,
        help="Show metadata for a given run_id (requires --run-root).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        help=(
            "Override number of parallel workers "
            "(TranspilerAgent parallel attempts)."
        ),
    )
    parser.add_argument(
        "--worker-processes",
        type=int,
        help="Override number of worker processes in worker manager.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Override max retries for each transpiler attempt.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable sandbox runner configuration (deny network, isolated).",
    )
    parser.add_argument(
        "--deny-network",
        action="store_true",
        help="Deny network access in sandboxed runs.",
    )
    parser.add_argument(
        "--run-timeout",
        type=int,
        help="Override sandbox run timeout (seconds).",
    )
    parser.add_argument(
        "--stream-mode",
        type=str,
        choices=["all", "winner", "none"],
        help="Console streaming mode (mirrors KernelAgent).",
    )
    parser.add_argument(
        "--resume-run",
        type=str,
        help="Resume an existing run (requires --run-root).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="Override base URL for OpenAI/Relay compatible providers.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        help="Override API key environment variable name for provider.",
    )
    parser.add_argument(
        "--high-reasoning",
        action="store_true",
        help=(
            "Request high reasoning effort for supported "
            "models (o1/o3/gpt-5)."
        ),
    )
    parser.add_argument(
        "--planner",
        type=str,
        help="Select an execution planner (e.g., kernel_agent_auto).",
    )
    parser.add_argument(
        "--plugin-module",
        action="append",
        dest="plugin_modules",
        help=(
            "Import the given module before planning "
            "(registers planners/delegates)."
        ),
    )
    return parser


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    return yaml.safe_load(config_path.read_text()) or {}


def _apply_cli_overrides(
    args: argparse.Namespace, config: dict, run_root: Path
) -> dict:
    trans_cfg = config.setdefault("transpilation", {})
    runtime_cfg = trans_cfg.setdefault("runtime", {})
    runtime_cfg["enabled"] = True
    runtime_cfg["run_root"] = str(run_root)
    if args.resume_run:
        runtime_cfg["run_id"] = args.resume_run
        runtime_cfg["resume"] = True

    llm_cfg = trans_cfg.setdefault("llm", {})
    if args.llm_provider:
        llm_cfg["provider"] = args.llm_provider
    if args.model:
        llm_cfg["model"] = args.model
    if args.api_base:
        llm_cfg["base_url"] = args.api_base
        llm_cfg["streaming"] = llm_cfg.get("streaming", {})
    if args.api_key_env:
        llm_cfg["api_key_env"] = args.api_key_env
    if args.high_reasoning:
        llm_cfg["high_reasoning_effort"] = True
    streaming_cfg = llm_cfg.setdefault("streaming", {})
    if args.stream_mode:
        streaming_cfg["mode"] = args.stream_mode
        streaming_cfg["enabled"] = args.stream_mode != "none"

    planner_cfg = trans_cfg.setdefault("planner", {})
    if args.planner:
        planner_cfg["kind"] = args.planner

    plugin_cfg = trans_cfg.setdefault("plugins", {})
    modules = plugin_cfg.setdefault("modules", [])
    if args.plugin_modules:
        for mod in args.plugin_modules:
            if mod not in modules:
                modules.append(mod)

    agents_cfg = trans_cfg.setdefault("agents", {})
    if args.parallel_workers:
        agents_cfg["parallel_workers"] = args.parallel_workers
    if args.max_retries:
        agents_cfg["max_retries"] = args.max_retries
    worker_mgr_cfg = agents_cfg.setdefault("worker_manager", {})
    if args.worker_processes:
        worker_mgr_cfg["enabled"] = True
        worker_mgr_cfg["workers"] = args.worker_processes

    verification_cfg = trans_cfg.setdefault("verification", {})
    if args.sandbox or args.deny_network or args.run_timeout:
        runner_cfg = verification_cfg.setdefault("runner", {})
        runner_cfg["kind"] = "sandbox"
        runner_cfg["run_root"] = str(run_root)
        if args.run_timeout:
            runner_cfg["timeout_s"] = args.run_timeout
        if args.sandbox:
            runner_cfg["isolated"] = True
        if args.deny_network:
            runner_cfg["deny_network"] = True

    return config


def _list_runs(run_root: Path) -> int:
    if not run_root.exists():
        print("No runs yet.", file=sys.stderr)
        return 0
    runs = sorted(run_root.iterdir())
    if not runs:
        print("No runs yet.", file=sys.stderr)
        return 0
    for run_dir in runs:
        session_file = run_dir / "session.json"
        summary = {"run_id": run_dir.name}
        if session_file.exists():
            try:
                summary.update(json.loads(session_file.read_text()))
            except Exception:
                pass
        print(json.dumps(summary, indent=2))
    return 0


def _show_run(run_root: Path, run_id: str) -> int:
    run_dir = run_root / run_id
    if not run_dir.exists():
        print(f"Run {run_id} not found under {run_root}", file=sys.stderr)
        return 1
    session_file = run_dir / "session.json"
    events = run_dir / "events.jsonl"
    units_dir = run_dir / "units"
    if session_file.exists():
        print(session_file.read_text())
    if events.exists():
        print("=== events ===")
        print(events.read_text())
    if units_dir.exists():
        print("=== units ===")
        for unit_file in sorted(units_dir.glob("*.json")):
            print(unit_file.read_text())
    return 0


def _import_plugin_modules(config: dict, config_root: Path) -> None:
    plugin_cfg = config.get("transpilation", {}).get("plugins", {}) or {}
    paths = plugin_cfg.get("paths") or []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (config_root / path).resolve()
        if not path.exists():
            print(f"Plugin path '{path}' does not exist", file=sys.stderr)
            continue
        for candidate in filter(None, {path, path.parent}):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
    modules = plugin_cfg.get("modules") or []
    for mod_name in modules:
        if not mod_name:
            continue
        try:
            import_module(mod_name)
        except Exception as exc:  # pragma: no cover
            print(
                f"Failed to import plugin module '{mod_name}': {exc}",
                file=sys.stderr,
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        load_dotenv()
    except Exception:
        pass

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )

    run_root = Path(args.run_root or ".transpile_runs")
    run_root.mkdir(parents=True, exist_ok=True)
    if args.list_runs:
        return _list_runs(run_root)
    if args.show_run:
        return _show_run(run_root, args.show_run)
    if args.resume_run and not args.run_root:
        parser.error("--resume-run requires --run-root")

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config_data = _load_config(config_path)
    config_data = _apply_cli_overrides(args, config_data, run_root)
    source, target = _resolve_source_and_target(
        args, config_data, config_path, run_root, parser
    )
    _import_plugin_modules(config_data, config_path.resolve().parent)

    planner_cfg = config_data.setdefault("transpilation", {}).setdefault(
        "planner", {}
    )
    planner = load_planner(planner_cfg.get("kind"), planner_cfg)
    plan: ExecutionPlan = planner.plan(source, config_data)
    if plan.context:
        print(f"[planner] {plan.context}", flush=True)
    if plan.action == "delegate":
        delegate_name = plan.context.get("delegate")
        delegate = load_delegate(delegate_name, config_data)
        try:
            exit_code = delegate.execute(
                source, target, run_root, plan, config_data
            )
        except Exception as exc:  # pragma: no cover
            print(f"Delegate execution failed: {exc}", file=sys.stderr)
            return 3
        if exit_code != 0:
            return exit_code
        if not args.no_verify:
            verify_code = _verify_delegate_output(
                delegate,
                source,
                target,
                config_data,
                config_root=config_path.resolve().parent,
            )
            if verify_code != 0:
                return verify_code
        print(f"Transpiled {source} -> {target}")
        print(f"Run artifacts stored in {run_root}")
        return exit_code

    orchestrator = TranspilationOrchestrator(config=config_data)

    verify = not args.no_verify
    orchestrator.transpile_file(source, target, verify=verify)
    print(f"Transpiled {source} -> {target}")
    print(f"Run artifacts stored in {run_root}")
    return 0


def _resolve_source_and_target(
    args: argparse.Namespace,
    config: dict,
    config_path: Path,
    run_root: Path,
    parser: argparse.ArgumentParser,
) -> tuple[Path, Path]:
    config_base = config_path.parent.resolve()
    settings = build_transpilation_settings(config, config_root=config_base)
    layout_cfg = config.setdefault("transpilation", {}).setdefault(
        "layout", {}
    )
    layout_output_cfg = layout_cfg.setdefault("output", {})
    layout = settings.layout

    if args.source:
        source = Path(args.source).expanduser()
    else:
        input_path = layout.input.path
        if not input_path:
            parser.error(
                "source path required unless layout.input.path is set."
            )
            return Path("."), Path(".")
        try:
            source = resolve_input_path(
                layout,
                config_base=config_base,
                run_root=run_root,
            )
        except ValueError as exc:
            parser.error(str(exc))
            return Path("."), Path(".")

    if not source.exists():
        parser.error(f"Source file '{source}' not found.")

    if args.target:
        target = Path(args.target).expanduser()
    else:
        target = resolve_output_path(
            layout,
            source=source,
            config_base=config_base,
            run_root=run_root,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    layout_output_cfg["resolved_path"] = str(target)
    layout_output_cfg.setdefault("kind", layout.output.kind)
    return source, target


def _verify_delegate_output(
    delegate,
    source_path: Path,
    target_path: Path,
    config: Dict[str, Any],
    *,
    config_root: Path,
) -> int:
    settings = build_transpilation_settings(config, config_root=config_root)
    context_builder = ContextBuilder()
    layout_plan = LayoutPlan(settings.layout_raw)
    ctx = context_builder.build(settings.integration, layout_plan)
    try:
        source_code = source_path.read_text(encoding="utf-8")
    except Exception:
        source_code = ""
    unit = TranspileUnit(
        id=source_path.stem or "delegate_unit",
        language=settings.source_language,
        source_code=source_code,
    )
    try:
        candidate_text = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"Delegate output missing: {exc}", file=sys.stderr)
        return 4
    candidate = CandidatePatchSet(files={target_path: candidate_text})
    try:
        result = delegate.verify(unit, candidate, ctx)
    except NotImplementedError:
        return 0
    except Exception as exc:  # pragma: no cover - unexpected delegate failure
        print(f"Delegate verification error: {exc}", file=sys.stderr)
        return 4
    if not result.passed:
        print(
            f"Result code failed verification with result: {result.details}",
            file=sys.stderr,
        )
        return 4
    print("Result code passed verification.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
