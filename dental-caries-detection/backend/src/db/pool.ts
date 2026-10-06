import pg from 'pg';

// One shared pool per process. Created lazily so code paths that never touch the
// DB (e.g. GET /health, unit tests with a mocked jobs module) don't need DATABASE_URL.
let pool: pg.Pool | null = null;

export function getPool(): pg.Pool {
  if (!pool) {
    pool = new pg.Pool({
      connectionString: process.env.DATABASE_URL,
      max: 5,
    });
    // An idle client erroring (e.g. Postgres restarted) must not crash the process;
    // the next query simply gets a fresh connection.
    pool.on('error', (err) => {
      console.error('[db] idle client error:', err.message);
    });
  }
  return pool;
}

export async function closePool(): Promise<void> {
  if (pool) {
    const p = pool;
    pool = null;
    await p.end();
  }
}
