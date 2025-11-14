# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications copyright (c) 2025 Logan Martel.
# Adapted from https://github.com/meta-pytorch/KernelAgent (Apache-2.0).

"""Worker manager adapted from KernelAgent's manager.py."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import shutil
import tempfile
import traceback

from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from langformer.worker.payload import WorkerPayload


class WorkerManager:
    def __init__(
        self,
        num_workers: int,
        worker_fn: Callable[[int, Dict[str, Any]], Dict[str, Any]],
        *,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.num_workers = num_workers
        self.worker_fn = worker_fn
        self.log_dir = Path(
            log_dir or tempfile.mkdtemp(prefix="transpile_workers_")
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workers_dir = self.log_dir / "workers"
        self.workers_dir.mkdir(exist_ok=True)
        try:
            self._ctx = mp.get_context("fork")
        except ValueError:  # pragma: no cover - windows fallback
            self._ctx = mp.get_context()
        self.success_event = self._ctx.Event()
        self.result_queue: MPQueue = self._ctx.Queue()
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def temp_workdirs(self) -> Iterator[List[Path]]:
        workdirs: List[Path] = []
        try:
            for i in range(self.num_workers):
                workdir = Path(tempfile.mkdtemp(prefix=f"worker_{i}_"))
                workdirs.append(workdir)
                self.logger.debug("Created workdir %s", workdir)
            yield workdirs
        finally:
            for workdir in workdirs:
                shutil.rmtree(workdir, ignore_errors=True)

    def run(
        self,
        tasks: Sequence[Union[WorkerPayload, Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        with self.temp_workdirs() as workdirs:
            processes: List[BaseProcess] = []
            for idx, (task, workdir) in enumerate(zip(tasks, workdirs)):
                if isinstance(task, WorkerPayload):
                    payload = task.to_dict()
                else:
                    payload = dict(task)
                worker_args = {
                    "workdir": workdir,
                    "log_dir": self.workers_dir / f"worker_{idx}",
                    **payload,
                }
                proc = self._ctx.Process(
                    target=_worker_entry,
                    args=(
                        self.worker_fn,
                        idx,
                        worker_args,
                        self.success_event,
                        self.result_queue,
                    ),
                )
                proc.start()
                processes.append(proc)
            winner = None
            while True:
                try:
                    result = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    if not any(p.is_alive() for p in processes):
                        break
                    continue
                if result.get("success"):
                    winner = result
                    self.success_event.set()
                    break
                self.logger.warning("Worker reported failure: %s", result)
            for proc in processes:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate()
                self.logger.debug(
                    "Worker %s exitcode=%s", proc.name, proc.exitcode
                )
            return winner


def _worker_entry(
    worker_fn,
    worker_id: int,
    inputs: Dict[str, Any],
    success_event: Event,
    result_queue: MPQueue,
) -> None:
    try:
        result = worker_fn(worker_id, inputs)
    except Exception as exc:  # pragma: no cover - diagnostics
        result_queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "variant": inputs.get("variant_label"),
            }
        )
        return
    if result.get("success"):
        result_queue.put(result)
    else:
        result_queue.put(result)
