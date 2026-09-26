"""
Inference runner — single-concurrency child-process manager.

Design:
  - Uses multiprocessing.get_context("spawn") to avoid fork-after-CUDA/OpenMP hazards.
  - Models are loaded inside the child process (not passed from parent).
  - A daemon monitor thread joins the child and marks the job 'fail' on crash.
  - Explicit cancellations are tracked in _cancelled_job_ids so the monitor
    does NOT mark the job 'fail' when the child is intentionally terminated.
  - cancel() escalates: SIGTERM → 10s join → SIGKILL if still alive.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from typing import Optional

from fastapi import HTTPException

log = logging.getLogger(__name__)

_ctx = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# Child process entry point (must be a module-level function for spawn pickling)
# ---------------------------------------------------------------------------

def _inference_process_entry(job_id: int, settings_data: dict) -> None:
    """
    Loaded fresh inside the child process.
    Imports settings and models here — nothing crosses the process boundary.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [child pid=%(process)d] %(name)s: %(message)s",
    )

    from app.config import Settings
    from models.registry import load_models_in_child
    from app.pipeline.orchestrator import run_pipeline

    settings = Settings.model_validate(settings_data)
    models = load_models_in_child(
        weights_dir=settings.weights_dir,
        device=settings.device,
        enable_crop_segmenter=settings.enable_crop_segmenter,
    )
    run_pipeline(job_id, settings, models)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class InferenceRunner:
    def __init__(self) -> None:
        self._process: Optional[multiprocessing.Process] = None
        self._active_job_id: Optional[int] = None
        self._cancelled_job_ids: set[int] = set()
        self._lock = threading.Lock()

    @property
    def active_job_id(self) -> Optional[int]:
        with self._lock:
            return self._active_job_id

    def start(self, job_id: int, settings_data: dict, db_engine: object) -> None:
        """
        Spawn a child process for the given job.
        Raises HTTP 409 if a job is already running.
        """
        from app.db import update_job_fail

        with self._lock:
            if self._process is not None and self._process.is_alive():
                raise HTTPException(
                    status_code=409,
                    detail="Inference already running. Call POST /cancel first.",
                )
            process = _ctx.Process(
                target=_inference_process_entry,
                args=(job_id, settings_data),
                daemon=False,  # must not be daemon — needs its own cleanup
                name=f"inference-job-{job_id}",
            )
            process.start()
            log.info("spawned inference child pid=%d for job_id=%d", process.pid, job_id)
            self._process = process
            self._active_job_id = job_id

        # Monitor thread — daemon so it doesn't block interpreter shutdown
        monitor = threading.Thread(
            target=self._monitor,
            args=(process, job_id, db_engine),
            daemon=True,
            name=f"monitor-job-{job_id}",
        )
        monitor.start()

    def cancel(self) -> None:
        """
        Idempotent cancellation.
        SIGTERM → join(10s) → SIGKILL if still alive.
        """
        with self._lock:
            if self._process is None or not self._process.is_alive():
                log.info("cancel: no active inference process")
                return
            cancelled_job_id = self._active_job_id
            self._cancelled_job_ids.add(cancelled_job_id)
            log.info(
                "cancelling inference pid=%d job_id=%d",
                self._process.pid, cancelled_job_id,
            )
            self._process.terminate()
            self._process.join(timeout=10)
            if self._process.is_alive():
                log.warning(
                    "process pid=%d did not stop after SIGTERM — sending SIGKILL",
                    self._process.pid,
                )
                self._process.kill()
                self._process.join()
            self._process = None
            self._active_job_id = None
            log.info("job_id=%d process terminated", cancelled_job_id)

    def _monitor(
        self,
        process: multiprocessing.Process,
        job_id: int,
        db_engine: object,
    ) -> None:
        """
        Wait for the child to finish.
        If it crashes (exit_code ≠ 0) and was NOT explicitly cancelled,
        mark the job as 'fail' in the database.
        """
        from app.db import update_job_fail

        process.join()
        exit_code = process.exitcode

        with self._lock:
            if self._process is process:
                self._process = None
                self._active_job_id = None

        was_cancelled = job_id in self._cancelled_job_ids
        self._cancelled_job_ids.discard(job_id)

        if exit_code in (0, None):
            log.info("inference child job_id=%d exited cleanly (exit_code=%s)", job_id, exit_code)
        elif was_cancelled:
            log.info(
                "inference child job_id=%d terminated by cancel (exit_code=%d) — no fail update",
                job_id, exit_code,
            )
        else:
            log.error(
                "inference child job_id=%d crashed (exit_code=%d) — marking fail",
                job_id, exit_code,
            )
            # Short message for DB; full stack trace already in child stderr/logs
            update_job_fail(
                db_engine,
                job_id,
                f"Inference process exited with code {exit_code}"[:255],
            )
