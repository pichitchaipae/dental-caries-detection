# Backend

Node.js orchestrator API (Fastify 5 + TypeScript, ESM). Owner: Naris.

## Layout (Sprint 1)

```
src/
├── app.ts               # buildApp(): Fastify instance + routes, no listen (used by tests)
├── server.ts            # Entry point: listen on BACKEND_PORT, SIGTERM/SIGINT shutdown
├── db/
│   └── migrate.ts       # Placeholder until BE-2.2; runnable via `npm run migrate`
└── __tests__/
    └── health.test.ts   # GET /health via app.inject()
```

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | `tsx watch src/server.ts` on port 8000 |
| `npm run build` | Compile to `dist/` (tests excluded via `tsconfig.build.json`) |
| `npm start` | Run the compiled server |
| `npm run migrate` | Run the compiled migration (`dist/db/migrate.js`) |
| `npm run typecheck` | Type-check `src/` including tests |
| `npm test` | Vitest |
| `npm run lint` | ESLint (flat config, type-aware `no-floating-promises`) |
| `npm run format` / `format:check` | Prettier |

Requires Node.js 22.12 or newer (Vitest 5 requirement).

## Docker

The image is multi-stage on `node:22-slim`, runs as `node`, and starts with
`node dist/db/migrate.js && exec node dist/server.js`. Build and run alone:

```bash
docker build -t caries-backend ./backend
docker run --rm -p 8000:8000 caries-backend
```
