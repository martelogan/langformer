"""Example script that runs the Langformer on a Python module."""

from __future__ import annotations

import argparse
import tempfile

from pathlib import Path

from langformer import TranspilationOrchestrator

SAMPLE_SOURCE = """\
def main(a: int = 1, b: int = 2) -> int:
    return a + b
"""


def ensure_source(path: Path | None) -> Path:
    if path is not None:
        return path
    temp_dir = Path(tempfile.mkdtemp(prefix="transpile-example-"))
    src = temp_dir / "sample.py"
    src.write_text(SAMPLE_SOURCE)
    return src


def main() -> None:
    parser = argparse.ArgumentParser(description="Transpile Python to Rust using the orchestrator")
    parser.add_argument("--config", type=Path, default=Path("configs/python_to_rust.yaml"))
    parser.add_argument("--source", type=Path, help="Python source file", default=None)
    parser.add_argument("--target", type=Path, help="Rust output file", default=Path(".transpile/output.rs"))
    parser.add_argument("--verify", action="store_true", help="Run verification (requires runnable target code)")
    args = parser.parse_args()

    source_path = ensure_source(args.source)
    target_path = args.target
    target_path.parent.mkdir(parents=True, exist_ok=True)

    orchestrator = TranspilationOrchestrator(config_path=args.config)
    orchestrator.transpile_file(source_path, target_path, verify=args.verify)
    print(f"Transpiled {source_path} -> {target_path}")


if __name__ == "__main__":
    main()
