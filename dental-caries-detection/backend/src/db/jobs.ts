/**
 * Job state helpers — the `jobs` table is the single source of truth for the
 * status of the latest submission (no in-memory state in the routes).
 *
 * Shared with ml-service, which only ever runs:
 *   UPDATE jobs SET status='done'|'fail', ... WHERE id=:job_id AND status='processing'
 * so a job this module marks 'superseded' can never be overwritten by the ML side,
 * and ml-service deletes its own result file when that UPDATE matches 0 rows.
 */

import { getPool } from './pool.js';

export type JobStatus = 'processing' | 'done' | 'fail' | 'superseded';

export interface Job {
  id: number;
  status: JobStatus;
  resultPath: string | null;
  failMessage: string | null;
}

// Arbitrary constant key for pg_advisory_xact_lock: serializes concurrent
// POST /process calls so "supersede old + insert new" is atomic.
const START_JOB_LOCK_KEY = 4_170_001;

interface JobRow {
  id: string; // BIGINT comes back as a string from node-postgres
  status: JobStatus;
  result_path: string | null;
  fail_message: string | null;
}

function toJob(row: JobRow): Job {
  return {
    id: Number(row.id),
    status: row.status,
    resultPath: row.result_path,
    failMessage: row.fail_message,
  };
}

/**
 * Start a new job: mark every still-processing job as 'superseded' and insert
 * the new one as 'processing', in one transaction.
 *
 * Rows are never deleted: ml-service treats a missing row as "no DB" and would
 * then keep a late result file instead of discarding it.
 *
 * The id is a millisecond timestamp (as before) but allocated under the lock and
 * forced above the current max, so "latest job" = highest id even when two
 * uploads race.
 *
 * @returns the new job id, and the ids of the jobs it replaced (the superseded
 *          ones plus the previous latest job) whose files can now be cleaned up.
 */
export async function startJob(
  now: number = Date.now()
): Promise<{ jobId: number; replacedJobIds: number[] }> {
  const client = await getPool().connect();
  try {
    await client.query('BEGIN');
    await client.query('SELECT pg_advisory_xact_lock($1)', [START_JOB_LOCK_KEY]);
    const superseded = await client.query<{ id: string }>(
      "UPDATE jobs SET status = 'superseded', updated_at = now() WHERE status = 'processing' RETURNING id"
    );
    const previous = await client.query<{ id: string }>(
      'SELECT id FROM jobs ORDER BY id DESC LIMIT 1'
    );
    const inserted = await client.query<{ id: string }>(
      `INSERT INTO jobs (id, status, updated_at)
       SELECT GREATEST($1::bigint, COALESCE(max(id), 0) + 1), 'processing', now() FROM jobs
       RETURNING id`,
      [now]
    );
    await client.query('COMMIT');
    const replaced = new Set([...superseded.rows, ...previous.rows].map((r) => Number(r.id)));
    return { jobId: Number(inserted.rows[0].id), replacedJobIds: [...replaced] };
  } catch (err) {
    await client.query('ROLLBACK').catch(() => {});
    throw err;
  } finally {
    client.release();
  }
}

/** The most recent submission, or null if no job was ever submitted. */
export async function getLatestJob(): Promise<Job | null> {
  const res = await getPool().query<JobRow>(
    'SELECT id, status, result_path, fail_message FROM jobs ORDER BY id DESC LIMIT 1'
  );
  return res.rows[0] ? toJob(res.rows[0]) : null;
}

/**
 * Mark a job as failed from the backend side (e.g. ml-service refused it).
 * Same guard as ml-service: only a job that is still 'processing' changes.
 */
export async function failJob(jobId: number, message: string): Promise<void> {
  await getPool().query(
    "UPDATE jobs SET status = 'fail', fail_message = $2, updated_at = now() WHERE id = $1 AND status = 'processing'",
    [jobId, message.slice(0, 255)]
  );
}
