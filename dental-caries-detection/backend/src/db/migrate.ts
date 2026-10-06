import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import pg from 'pg';

export async function migrate(): Promise<void> {
  const { Client } = pg;
  const client = new Client({
    connectionString: process.env.DATABASE_URL,
  });

  try {
    await client.connect();
    // Idempotent: safe to run on every start and on databases created by the
    // earlier version of this file. Column names/types used by ml-service
    // (id, status, result_path, fail_message, updated_at) must not change.
    await client.query(`
      CREATE TABLE IF NOT EXISTS jobs (
        id BIGINT PRIMARY KEY,
        status VARCHAR(255) NOT NULL DEFAULT 'processing',
        result_path VARCHAR(255),
        fail_message VARCHAR(255),
        updated_at TIMESTAMP,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
      );

      ALTER TABLE jobs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();

      DO $$
      BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'jobs_status_check') THEN
          ALTER TABLE jobs ADD CONSTRAINT jobs_status_check
            CHECK (status IN ('processing', 'done', 'fail', 'superseded'));
        END IF;
      END $$;

      -- The previous backend never superseded preempted jobs, so older rows can
      -- still say 'processing'. Keep only the newest one active before adding
      -- the one-active-job index.
      UPDATE jobs SET status = 'superseded', updated_at = now()
       WHERE status = 'processing'
         AND id <> (SELECT max(id) FROM jobs WHERE status = 'processing');

      CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs ((true)) WHERE status = 'processing';
    `);
    console.info('migration: jobs table is up to date');
  } catch (err) {
    console.error('migration error:', err);
    throw err;
  } finally {
    await client.end();
  }
}

const isMain =
  process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) {
  try {
    await migrate();
  } catch (err) {
    console.error('migration failed', err);
    process.exit(1);
  }
}
