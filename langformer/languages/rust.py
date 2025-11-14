"""Rust language plugin that shells out to `rustc`."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Any, Dict, Optional

from langformer.languages.base import LanguagePlugin


class LightweightRustLanguagePlugin(LanguagePlugin):
    """Minimal Rust plugin relying on the system `rustc`."""

    language_name = "rust"

    def parse(self, source_code: str) -> str:
        return source_code

    def compile(self, code: str) -> bool:
        tmpdir = Path(tempfile.mkdtemp(prefix="transpile-rust-"))
        try:
            self._invoke_rustc(code, tmpdir, emit_metadata=True)
            return True
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def execute(
        self, code: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        tmpdir = Path(tempfile.mkdtemp(prefix="transpile-rust-"))
        inputs = inputs or {}
        args = inputs.get("args", [])
        env = os.environ.copy()
        env.update(inputs.get("env", {}))
        try:
            binary_path = self._invoke_rustc(code, tmpdir, emit_metadata=False)
            proc = subprocess.run(
                [str(binary_path), *map(str, args)],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            return {
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _invoke_rustc(
        self, code: str, tmpdir: Path, *, emit_metadata: bool
    ) -> Path:
        src_path = tmpdir / "transpile.rs"
        bin_path = tmpdir / "transpile_bin"
        src_path.write_text(code)
        args = ["rustc", "--edition=2021", str(src_path)]
        if emit_metadata:
            args.append("--emit=metadata")
        else:
            args.extend(["-o", str(bin_path)])
        subprocess.run(args, check=True, capture_output=True)
        return bin_path


__all__ = ["LightweightRustLanguagePlugin"]
