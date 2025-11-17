"""CLI entry point for the DSPy Python→Ruby example."""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

import yaml

from dotenv import load_dotenv

DEFAULT_CONFIG = Path("configs/dspy_py2rb.yaml")
DEFAULT_SOURCE = Path(
    "examples/dspy_py2rb_transpiler/inputs/report_module.py"
)
DEFAULT_OUTPUT_DIR = Path("examples/dspy_py2rb_transpiler/outputs")


def _ensure_project_root() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    """Run the DSPy transpilation example."""

    _ensure_project_root()
    from examples.dspy_py2rb_transpiler.oracle import register_oracle
    from langformer import TranspilationOrchestrator
    parser = argparse.ArgumentParser(
        description="DSPy-powered Python→Ruby transpilation example"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Configuration file to use",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Python source module to transpile",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional output path for the generated Ruby file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the configured oracle/verification strategy",
    )
    args = parser.parse_args()

    load_dotenv()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
    register_oracle()

    config_data = yaml.safe_load(args.config.read_text())
    orchestrator = TranspilationOrchestrator(
        config_path=args.config,
        config=config_data,
    )
    source_path = Path(args.source)
    target_path = _resolve_target_path(source_path, args.target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator.transpile_file(source_path, target_path, verify=args.verify)
    logging.info(
        "DSPy transpilation complete: %s → %s", source_path, target_path
    )


def _resolve_target_path(source: Path, explicit_target: Path | None) -> Path:
    if explicit_target is not None:
        return Path(explicit_target)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{source.stem}.rb"


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
