"""Ruby language plugin that shells out to the Ruby interpreter."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Any, Dict, Optional

from langformer.languages.base import LanguagePlugin


class LightweightRubyLanguagePlugin(LanguagePlugin):
    """Executes Ruby code via the system `ruby` command."""

    language_name = "ruby"

    def parse(self, source_code: str) -> str:
        return source_code

    def compile(self, code: str) -> bool:
        temp_dir = Path(tempfile.mkdtemp(prefix="transpile-ruby-"))
        try:
            src_path = temp_dir / "transpile.rb"
            src_path.write_text(code)
            subprocess.run(
                ["ruby", "-c", str(src_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def execute(
        self, code: str, inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        temp_dir = Path(tempfile.mkdtemp(prefix="transpile-ruby-"))
        inputs = inputs or {}
        args = inputs.get("args", [])
        env = inputs.get("env")
        try:
            src_path = temp_dir / "transpile.rb"
            src_path.write_text(code)
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)
            proc = subprocess.run(
                ["ruby", str(src_path), *map(str, args)],
                capture_output=True,
                text=True,
                env=exec_env,
                check=True,
            )
            return {
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = ["LightweightRubyLanguagePlugin"]
