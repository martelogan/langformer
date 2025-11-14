"""CLI helper that transpiles the sample Python module to Ruby."""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

import yaml

from dotenv import load_dotenv

from langformer import TranspilationOrchestrator

DEFAULT_CONFIG = Path("configs/simple_py2rb.yaml")
DEFAULT_SOURCE = Path(
    "examples/simple_py2rb_transpiler/inputs/sample_module.py"
)
DEFAULT_OUTPUT_DIR = Path("examples/simple_py2rb_transpiler/outputs/generated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple Python→Ruby transpilation example"
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
        help=(
            "Destination for the generated Ruby file "
            "(defaults to outputs/<source-stem>.rb)"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Run the configured verification strategy "
            "(requires Ruby runtime)"
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )

    config_data = yaml.safe_load(args.config.read_text())
    orchestrator = TranspilationOrchestrator(
        config_path=args.config,
        config=config_data,
    )
    source_path = Path(args.source)
    target_path = _resolve_target_path(source_path, args.target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator.transpile_file(source_path, target_path, verify=args.verify)
    logging.info("Transpiled %s → %s", source_path, target_path)


def _resolve_target_path(source: Path, explicit_target: Path | None) -> Path:
    if explicit_target is not None:
        return Path(explicit_target)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{source.stem}.rb"


if __name__ == "__main__":
    main()
