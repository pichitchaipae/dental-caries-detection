"""
Database helpers — update-only (ML service never inserts rows).

All updates are scoped to  WHERE id = :job_id AND status = 'processing'
so a superseded job can never accidentally overwrite the current job's status.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger(__name__)

_MAX_MSG_LEN = 255


def get_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with a small pool appropriate for a single-worker service."""
    return create_engine(
        database_url,
        pool_size=2,
        max_overflow=0,
        pool_pre_ping=True,
    )


def update_job_done(engine: Engine, job_id: int, result_path: str) -> bool:
    """
    Set status='done' and result_path for the given job.

    Returns True if the row was actually updated (i.e. job was still 'processing').
    Returns False if the job was already superseded or in another terminal state.
    """
    sql = text(
        """
        UPDATE jobs
           SET status      = 'done',
               result_path = :result_path,
               updated_at  = now()
         WHERE id     = :job_id
           AND status = 'processing'
        """
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {"job_id": job_id, "result_path": result_path})
            updated = result.rowcount == 1
        if not updated:
            log.warning("update_job_done: job_id=%d not updated (likely superseded)", job_id)
        return updated
    except Exception:
        log.exception("update_job_done: DB error for job_id=%d", job_id)
        return False


def update_job_fail(engine: Engine, job_id: int, message: str) -> bool:
    """
    Set status='fail' with a human-readable fail_message (truncated to 255 chars).

    Returns True if the row was actually updated.
    """
    short_msg = message[:_MAX_MSG_LEN]
    sql = text(
        """
        UPDATE jobs
           SET status       = 'fail',
               fail_message = :msg,
               updated_at   = now()
         WHERE id     = :job_id
           AND status = 'processing'
        """
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(sql, {"job_id": job_id, "msg": short_msg})
            updated = result.rowcount == 1
        if not updated:
            log.warning("update_job_fail: job_id=%d not updated (likely superseded)", job_id)
        return updated
    except Exception:
        log.exception("update_job_fail: DB error for job_id=%d", job_id)
        return False


def get_job_status(engine: Engine, job_id: int) -> Optional[str]:
    """
    Return the current status of the job, or None if the job does not exist.
    """
    sql = text("SELECT status FROM jobs WHERE id = :job_id")
    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"job_id": job_id}).fetchone()
        return row[0] if row else None
    except Exception:
        log.exception("get_job_status: DB error for job_id=%d", job_id)
        return None
