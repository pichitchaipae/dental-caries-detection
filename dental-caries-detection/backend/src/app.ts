import Fastify, { type FastifyInstance, type FastifyServerOptions } from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import { registerProcessRoutes } from './routes/process.js';

// Builds the Fastify instance without listening, so tests can use app.inject().
export function buildApp(opts: FastifyServerOptions = {}): FastifyInstance {
  const app = Fastify(opts);

  app.register(cors, {
    origin: process.env.CORS_ORIGIN ?? '*',
  });
  
  app.register(multipart, {
    limits: {
      fileSize: 10 * 1024 * 1024 // 10MB limit
    }
  });

  // Backend liveness only; the ML service gets its own /health in Phase 2.
  // TODO BE-2.4: deep variant { status, db } via healthcheckDb().
  app.get('/health', () => ({ status: 'ok' }));

  app.register(registerProcessRoutes);

  return app;
}
