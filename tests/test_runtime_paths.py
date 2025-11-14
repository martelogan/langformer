from pathlib import Path

import pytest

from langformer.runtime import config, paths


def test_new_run_id_unique():
    run_id1 = config.new_run_id()
    run_id2 = config.new_run_id()
    assert run_id1 != run_id2


def test_make_run_dirs(tmp_path: Path):
    run_id = "run_test"
    dirs = paths.make_run_dirs(tmp_path, run_id)
    assert dirs.orchestrator == (tmp_path / run_id / "orchestrator")
    assert dirs.workers.exists()


def test_ensure_abs_regular_file(tmp_path: Path):
    file_path = tmp_path / "foo.txt"
    file_path.write_text("hello")
    assert paths.ensure_abs_regular_file(file_path) == file_path
    with pytest.raises(paths.PathSafetyError):
        paths.ensure_abs_regular_file("relative.txt")
