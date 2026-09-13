import Fastify, { type FastifyInstance, type FastifyServerOptions } from 'fastify';

// Builds the Fastify instance without listening, so tests can use app.inject().
export function buildApp(opts: FastifyServerOptions = {}): FastifyInstance {
  const app = Fastify(opts);

  // Backend liveness only; the ML service gets its own /health in Phase 2.
  // TODO BE-2.4: deep variant { status, db } via healthcheckDb().
  app.get('/health', () => ({ status: 'ok' }));

  // TODO BE-2.5: register @fastify/cors restricted to CORS_ORIGIN.
  // TODO BE-3.6 / BE-3.7: register routes/process.ts (POST /process, GET /process).

  return app;
}
