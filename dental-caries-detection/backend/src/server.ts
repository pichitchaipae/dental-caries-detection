import { buildApp } from './app.js';

const port = Number(process.env.BACKEND_PORT ?? 8000);
const host = '0.0.0.0';

const app = buildApp({ logger: true });

// TODO BE-2.4: await migrate() (with retry) before listen(); close the pool in onClose.

// Node running as PID 1 in a container ignores SIGTERM unless a handler exists,
// which would make `docker compose stop` wait for the 10s kill timeout.
for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.once(signal, () => {
    app.log.info({ signal }, 'shutting down');
    app.close().then(
      () => process.exit(0),
      (err: unknown) => {
        app.log.error(err, 'error during shutdown');
        process.exit(1);
      }
    );
  });
}

try {
  await app.listen({ port, host });
} catch (err) {
  app.log.error(err, 'failed to start server');
  process.exit(1);
}
