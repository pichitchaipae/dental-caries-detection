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
    await client.query(`
      CREATE TABLE IF NOT EXISTS jobs (
        id BIGINT PRIMARY KEY,
        status VARCHAR(255) NOT NULL DEFAULT 'processing',
        result_path VARCHAR(255),
        fail_message VARCHAR(255),
        updated_at TIMESTAMP
      );
    `);
    console.info('migration: jobs table created successfully');
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
