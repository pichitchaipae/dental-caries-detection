import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Placeholder until BE-2.2. It must not require a database, so the container
// entrypoint (`migrate && server`) still boots when no `db` is reachable.
export async function migrate(): Promise<void> {
  console.info('migration: placeholder, nothing to do (implemented in BE-2.2)');
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
