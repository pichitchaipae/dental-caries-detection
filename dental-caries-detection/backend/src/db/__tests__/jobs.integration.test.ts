import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'vitest';

// Runs against a real Postgres only when TEST_DATABASE_URL is set, e.g.
//   docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=pw --name be-test-db postgres:16
//   TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/postgres npm test
const url = process.env.TEST_DATABASE_URL;

describe.skipIf(!url)('jobs table (integration)', async () => {
  process.env.DATABASE_URL = url;
  const { migrate } = await import('../migrate.js');
  const { getPool, closePool } = await import('../pool.js');
  const { startJob, getLatestJob, failJob } = await import('../jobs.js');

  // The exact statement ml-service runs (ml-service/app/db.py update_job_done).
  const mlMarkDone = (id: number) =>
    getPool().query(
      "UPDATE jobs SET status='done', result_path=$2, updated_at=now() WHERE id=$1 AND status='processing'",
      [id, `/shared/result-${id}.json`]
    );

  beforeAll(async () => {
    await getPool().query('DROP TABLE IF EXISTS jobs');
    await migrate();
  });

  beforeEach(async () => {
    await getPool().query('TRUNCATE jobs');
  });

  afterAll(async () => {
    await getPool().query('DROP TABLE IF EXISTS jobs');
    await closePool();
  });

  it('migration is idempotent', async () => {
    await migrate();
    await migrate();
  });

  it('upgrades a table created by the previous migration', async () => {
    await getPool().query('DROP TABLE jobs');
    await getPool().query(`
      CREATE TABLE jobs (id BIGINT PRIMARY KEY, status VARCHAR(255) NOT NULL DEFAULT 'processing',
        result_path VARCHAR(255), fail_message VARCHAR(255), updated_at TIMESTAMP);
      INSERT INTO jobs (id, status) VALUES (1, 'processing'), (2, 'processing'), (3, 'done');
    `);
    await migrate();
    const rows = await getPool().query('SELECT id, status, created_at FROM jobs ORDER BY id');
    expect(rows.rows.map((r) => [Number(r.id), r.status])).toEqual([
      [1, 'superseded'],
      [2, 'processing'],
      [3, 'done'],
    ]);
    expect(rows.rows[0].created_at).toBeInstanceOf(Date);
  });

  it('getLatestJob is null on an empty table', async () => {
    expect(await getLatestJob()).toBeNull();
  });

  it('startJob supersedes the running job and returns replaced ids', async () => {
    expect(await startJob(100)).toEqual({ jobId: 100, replacedJobIds: [] });
    expect(await startJob(200)).toEqual({ jobId: 200, replacedJobIds: [100] });
    expect(await getLatestJob()).toMatchObject({ id: 200, status: 'processing' });
    const old = await getPool().query('SELECT status FROM jobs WHERE id = 100');
    expect(old.rows[0].status).toBe('superseded');
  });

  it('a superseded job cannot be marked done by ml-service', async () => {
    await startJob(100);
    await startJob(200);
    const late = await mlMarkDone(100);
    expect(late.rowCount).toBe(0);
    const current = await mlMarkDone(200);
    expect(current.rowCount).toBe(1);
    expect(await getLatestJob()).toEqual({
      id: 200,
      status: 'done',
      resultPath: '/shared/result-200.json',
      failMessage: null,
    });
  });

  it('returns the previous finished job as replaced too', async () => {
    await startJob(100);
    await mlMarkDone(100);
    expect((await startJob(200)).replacedJobIds).toEqual([100]);
  });

  it('failJob only touches a processing job and truncates the message', async () => {
    await startJob(100);
    await failJob(100, 'x'.repeat(300));
    const job = await getLatestJob();
    expect(job?.status).toBe('fail');
    expect(job?.failMessage).toHaveLength(255);

    await mlMarkDone(100); // no effect: not processing any more
    expect((await getLatestJob())?.status).toBe('fail');
  });

  it('allocates increasing ids even when the clock value is not', async () => {
    await startJob(500);
    expect((await startJob(400)).jobId).toBe(501);
  });

  it('concurrent submissions leave exactly one processing job, and it is the latest', async () => {
    const started = await Promise.all([startJob(1), startJob(1), startJob(1), startJob(1)]);
    expect(new Set(started.map((s) => s.jobId)).size).toBe(4);
    const active = await getPool().query("SELECT id FROM jobs WHERE status = 'processing'");
    expect(active.rowCount).toBe(1);
    expect((await getLatestJob())?.status).toBe('processing');
  });

  it('rejects unknown statuses', async () => {
    await expect(
      getPool().query("INSERT INTO jobs (id, status) VALUES (9, 'idle')")
    ).rejects.toThrow(/jobs_status_check/);
  });
});
