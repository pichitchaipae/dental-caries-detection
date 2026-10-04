# Project Note — Dental Caries Surface Classification

> Maintained by Claude as the primary context reference for this workspace.
> Last updated: 2026-09-20 (repo state as of commit `d531434`, plus uncommitted
> frontend Sprint 1 scaffold — see Section 3).

## 1. What this project is

A locally-deployed (Docker Compose, single clinic machine) AI inference tool.
A clinician uploads one panoramic dental X-ray (OPG); the system detects each
tooth, aligns it, classifies caries per anatomical surface, and shows the
result in an interactive Canvas viewer. The system is **stateless with
respect to patient data** — PostgreSQL stores only transient job-processing
status (one row per submission), never images or patient identifiers.

Full architecture, API contracts, DB schema, and design rationale live in
`dental-caries-detection/docs-md/project-structure.md` (the source of truth —
do not duplicate its content here, just reference it).

The repo root also contains a large, mostly-unrelated `reserch/` directory:
prior ML research/model-training notebooks and reports for the underlying
caries-classification model. That work is not part of the Phase 1 app build
tracked below; treat it as background/reference material only unless a task
explicitly calls for it.

## 2. Tech stack

| Layer | Technology | Status |
|---|---|---|
| Frontend | React 18 + TypeScript + Vite, HTML5 Canvas rendering | Sprint 1 (bootstrap) done and Docker-verified; UI/domain work not started |
| Backend | Node.js 22 + Fastify 5 + TypeScript (ESM), `pg`, `zod` | Sprint 1 (bootstrap) done; DB/API layers not started |
| ML tier (Phase 1) | In-process **Mock ML driver** inside the backend (no Python service) | Not started |
| ML tier (Phase 2, deferred) | Python 3 + FastAPI + Detectron2 + PCA + scikit-learn | Explicitly out of scope for Phase 1; `ml-service/` must not exist yet |
| Database | PostgreSQL 16 | Container wired up in Compose; schema/migration still a placeholder |
| DB admin | pgAdmin 4 | Working (dev-only, `127.0.0.1:5050`) |
| Orchestration | Docker Compose (`db`, `pgadmin`, `backend`, `frontend`) | **All four services confirmed healthy** via `docker compose up -d` (2026-09-20) |

Node target: `node:22-slim`. Requires Node ≥ 22.12 locally (Vitest 5
requirement) per `backend/README.md`.

## 3. Current implementation status

Tracked against the Phase 1 plan in
`dental-caries-detection/docs-md/task-pm-phase1.md` (7 sprints, IDs like
`BE-x.x` / `FE-x.x`). Corrections the team already found when actually
implementing Sprint 1 are written up in
`dental-caries-detection/docs-md/be-sprint1-report.md` — read that before
trusting `task-pm-phase1.md` literally on anything Sprint-1-related
(healthchecks, Dockerfile entrypoint, `.env` keys, DB schema shape, etc. all
changed from what the plan originally said).

### Done
- **Sprint 1 — Project init & Docker Compose skeleton: complete on both sides
  and Docker-verified end-to-end** (`db`, `pgadmin`, `backend`, `frontend` all
  `healthy` via `docker compose up -d`). Frontend files (`FE-1.4`, `FE-1.5`,
  `FE-1.7`) are currently **uncommitted** on Sukollapat's machine — not yet
  pushed/PR'd.
  - `frontend/`: Vite + React 18 + TS bootstrap, placeholder `<h1>` app,
    multi-stage `node:22-slim` Dockerfile, flat ESLint config + Prettier
    matching backend's conventions. Two non-obvious Docker-only bugs were
    found and fixed while verifying against a live daemon (neither reproduces
    outside a container, so watch for regressions if the Dockerfile changes):
    1. `vite preview` bundles `vite.config.ts` to a temp file inside `/app` on
       every start; `/app` is root-owned from `COPY`, so the container's
       unprivileged `node` user got `EACCES`. Fixed with
       `RUN chown -R node:node /app` before `USER node` (same pattern as
       backend's `/shared` fix).
    2. `vite.config.ts` must not `import` from `'vitest/config'` (only
       `/// <reference types="vitest/config" />`) — `vitest` is a
       devDependency stripped from the runtime image, but `vite preview`
       still executes this file, so a real import throws
       `ERR_MODULE_NOT_FOUND`.
    3. `vite` and `@vitejs/plugin-react` must be regular `dependencies`, not
       `devDependencies` — forced by `project-structure.md` §11.8 (serve via
       `vite preview`, no Nginx), since the runtime image excludes dev deps.
    4. Plan doc says `vite@5`, but `vitest@5.0.0` (already pinned by backend)
       requires `vite ^6.4.0+` — used `vite@6.4.3` + `@vitejs/plugin-react@5.2.0`.
- **Sprint 1 — Project init & Docker Compose skeleton (backend side): complete.**
  - Monorepo scaffold, `.env.example`, root `README.md`, `.gitignore` (BE-1.1)
  - Fastify + TypeScript bootstrap: `backend/src/app.ts` (`buildApp()`, no
    `listen()`, used by tests) + `backend/src/server.ts` (entry point, graceful
    `SIGTERM`/`SIGINT` shutdown). `GET /health` returns `{"status":"ok"}` only
    — no DB check yet (BE-1.2)
  - Backend Dockerfile: multi-stage on `node:22-slim`, runs as `node` user,
    entrypoint `node dist/db/migrate.js && exec node dist/server.js` (BE-1.3)
  - `docker-compose.yml`: `db`, `pgadmin`, `backend` all healthy;
    `POSTGRES_HOST_PORT` made configurable (default 5432) to avoid clashing
    with a local Postgres install; Node-`fetch`-based healthchecks (image has
    no `wget`/`curl`) (BE-1.6)
  - ESLint flat config + Prettier, `no-floating-promises` enforced (BE-1.7)
- `backend/src/db/migrate.ts` exists but is an intentional **no-op
  placeholder** — logs a message and exits cleanly, so the container boots
  even with no DB reachable. Real schema work is Sprint 2 (BE-2.2).

### Not started
- **Sprint 2 (BE-2.x)** — DB connection pool (`db/pool.ts`), idempotent
  migration creating the `jobs` table, job state data-access helpers
  (`db/jobs.ts`), wiring migration + pool shutdown into `server.ts`, CORS
  registration (`@fastify/cors` is already a dependency but unused).
- **Sprint 3 (BE-3.x)** — image validation, image store (`/shared` volume
  read/write + base64 encoding), `MlDriver` interface + Mock ML driver
  (`mockMl.ts`), single-flight process control, `POST /process` /
  `GET /process` routes, integration tests, and the `INT-1`/`INT-2` API
  contract + `result.sample.json` fixture (`backend/src/fixtures/`) — **does
  not exist yet**, and is a hard blocker for `FE-4.2`/`FE-4.3` below.
- **Sprint 4–7 (FE-x.x) beyond the Sprint 1 bootstrap** — layout shell
  (`FE-4.1`, unblocked, safe to start), domain types / API client
  (`FE-4.2`/`FE-4.3`, blocked on the `INT-2` fixture above), upload/polling
  workflow, Canvas viewer, detail panels.
- `ml-service/` correctly does not exist (deferred to Phase 2, per plan).

### ⚠️ Known plan/doc inconsistency to watch for
`project-structure.md` (Sept 9) describes the `jobs` table as a **history
table** (`SERIAL` id, `superseded` status, `one_active_job` unique index,
per-id updates) that supersedes on preemption. `task-pm-phase1.md` (Aug 30)
still describes a **singleton row** (`status='idle'` seed row, in-place
update, `p-limit(1)` concurrency guard, no `superseded` state). These are
incompatible schemas. `be-sprint1-report.md` §3.3 documents the conflict in
detail but does not resolve it. **Before starting Sprint 2 (BE-2.2/2.3), this
needs an explicit decision from the user/team on which model to implement** —
this is exactly the kind of ambiguity that should be raised rather than
guessed at.

## 4. Next logical steps (not yet assigned/confirmed by user)
1. Commit/PR the frontend Sprint 1 scaffold (currently uncommitted).
2. Resolve the jobs-table schema conflict above.
3. Sprint 2: DB pool, migration, job data-access layer.
4. Sprint 3: validation, image store, Mock ML driver, process control, the two
   HTTP routes, and the `INT-1`/`INT-2` API contract + fixture (needed before
   FE can start `FE-4.2`/`FE-4.3`).
5. `FE-4.1` (layout shell) can start any time — no backend dependency.

## 5. Key file map
- `dental-caries-detection/docker-compose.yml` — all 4 services confirmed working
- `dental-caries-detection/.env` — machine-local, gitignored; `POSTGRES_HOST_PORT` may need bumping past 5432/5433 on machines already running other Postgres containers (this dev machine needed 5434)
- `dental-caries-detection/.env.example` — all Phase 1 config keys, comments kept on their own line (env-file parsing gotcha, see be-sprint1-report.md #12)
- `dental-caries-detection/backend/src/app.ts` / `server.ts` / `db/migrate.ts` — current backend code, all pre-Sprint-2
- `dental-caries-detection/frontend/` — Sprint 1 scaffold (uncommitted), see Section 3 for the two Docker-only bugs found and fixed
- `dental-caries-detection/docs-md/project-structure.md` — architecture/API/DB source of truth
- `dental-caries-detection/docs-md/task-pm-phase1.md` — sprint/task breakdown (has known-stale parts, see report)
- `dental-caries-detection/docs-md/be-sprint1-report.md` — corrections found during actual Sprint 1 implementation (in Thai)
- `reserch/` — prior ML research/training work, not part of the Phase 1 app
