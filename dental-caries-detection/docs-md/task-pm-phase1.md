> **Last Updated:** 2026-08-30 22:56:06 +07
> **Document Owner:** Project Manager / Technical Lead
> **Audience:** Frontend Developer (FE), Backend / Node.js Developer (BE)
> **Related Documents:** [`zz-claude/project-structure.md`](project-structure.md)

# Phase 1 - Project Management & Task Delegation Plan

## Dental Caries Surface Classification - FE + BE Foundation (No ML)

---

## 0. How To Read This Document

Every task below is written as a self-contained work item with the following fields:

| Field | Meaning |
|---|---|
| **ID** | Stable identifier (e.g. `BE-1.2`). Use it in commit messages and stand-ups. |
| **Task & Objective** | What to build and why it matters. |
| **Assignee** | `FE` (Frontend Developer) or `BE` (Backend / Node.js Developer). |
| **Target Path** | Exact folder/file from the repository tree in `project-structure.md` Section 9. New files not yet in the tree are marked **(new)**. |
| **Technical Requirements** | Concrete implementation contract: props, payloads, schema, function signatures. |
| **Acceptance Criteria (DoD)** | Binary, testable conditions. The task is not "done" until every box is checked. |

**Convention:** `[ ]` = not started, `[~]` = in progress, `[x]` = done and reviewed.

---

## 1. Phase 1 Scope Statement

### 1.1 In Scope (Phase 1)

- Monorepo initialization, tooling, and one-command Docker Compose bring-up for **four** services: `db`, `pgadmin`, `backend`, `frontend`.
- **Backend (Node.js / Fastify / TypeScript):** full HTTP API surface (`POST /process`, `GET /process`), PostgreSQL schema ownership and idempotent migration, authoritative image validation, shared-volume image storage and base64 encoding, single-flight process control, and a **Mock ML driver** that stands in for the FastAPI ML service.
- **Frontend (React 18 / TypeScript / Vite):** SPA shell, typed API client, OPG upload with pre-flight validation, polling workflow, interactive HTML5 Canvas result viewer (bounding boxes, masks, PCA axes, FDI labels), per-tooth detail panel, and findings table.
- End-to-end happy-path and failure-path demo driven entirely by the Mock ML driver.

### 1.2 Explicitly OUT of Scope (Deferred to Phase 2)

| Deferred item | Reason |
|---|---|
| `ml-service/` directory and all Python / FastAPI / Detectron2 / PCA / classifier code | ML integration is a dedicated later phase. |
| Real model weights, `weights/`, `models/registry.py`, `pipeline/*` | Same as above. |
| `docker/` ML image build, GPU runtime, `requirements.txt` for ML | Same as above. |
| Real `POST /infer` / `POST /cancel` / `GET /health` HTTP calls to a live ML container | Replaced in Phase 1 by the Mock ML driver (`BE-4.x`). |

> **Note on the ML service container:** In Phase 1 the Compose file MUST NOT require `ml-service` to be running. The backend reaches the ML tier only through `services/mlClient.ts`, which in Phase 1 is wired to the in-process Mock ML driver via an environment flag (`ML_DRIVER=mock`). Phase 2 will flip `ML_DRIVER=http` and add the container with **zero changes** to routes, DB, or frontend.

### 1.3 Authentication - Deliberately NOT Built

The architecture (`project-structure.md` Sections 1.1, 3, 11.8, 12.2) specifies a **single-clinician, single-machine, localhost-only** deployment with **no user accounts and no patient data**. Therefore **Phase 1 contains no login, no session, no JWT, and no user table.** The only "security" work is CORS restriction and upload limits, captured in `BE-2.5` and `BE-6.3`. Any sprint that an outside template would label "Authentication" is here labelled **"Core UI Shell & App State"** and contains no auth.

---

## 2. Team Roles & Folder Responsibilities

### 2.1 Ownership Map

| Directory / File (from `project-structure.md` Section 9) | Owner | Notes |
|---|---|---|
| `frontend/` (entire subtree) | **FE** | SPA, Canvas viewer, API client, validation mirror. |
| `frontend/Dockerfile`, `frontend/vite.config.ts` | **FE** | Build + `vite preview` serving on port 3000. |
| `backend/` (entire subtree) | **BE** | Fastify API, DB schema, migration, process control, image store. |
| `backend/Dockerfile` | **BE** | `node:20-slim`, runs migration then starts Fastify. |
| `backend/src/services/mockMl.ts` **(new)** | **BE** | Phase 1 stand-in for the ML service. |
| `docker-compose.yml` | **BE** (primary) / FE (reviews `frontend` service block) | BE owns the file; FE must approve the `frontend` service, ports, and env passthrough. |
| `.env.example` | **BE** (primary) / FE (adds `FRONTEND_*` keys) | Shared config template. |
| `README.md` | **Shared** | BE writes "Run the stack"; FE writes "Frontend dev notes". |
| `ml-service/` | **Nobody in Phase 1** | Do not create. Placeholder only if Compose needs a stub (it does not). |
| `zz-claude/` | PM / Tech Lead | Planning docs, logs. |

### 2.2 Shared Contracts (neither developer changes these unilaterally)

| Contract artifact | Location | Change process |
|---|---|---|
| Frontend-facing API shape (`POST /process`, `GET /process`) | `project-structure.md` Section 7.1 + `INT-1` sign-off | Any change requires a joint 15-min sync and an update to both `backend/src/routes/process.ts` and `frontend/src/domain/inference.ts` in the **same PR pair**. |
| `result.json` schema (the `data` object) | `INT-2` fixture file | Fixture file is the single source of truth; both sides import/validate against it. |
| `jobs` table schema | `project-structure.md` Section 8 + `BE-1.4` | BE owns; announce in channel before altering. |
| Shared volume file names (`input.jpg`, `result.json`) and mount path (`/shared`) | `project-structure.md` Section 4.7 | Fixed. Do not rename. |

### 2.3 Working Agreement

- **Branch naming:** `feat/be-1.2-fastify-bootstrap`, `feat/fe-3.1-vite-setup`.
- **PR rule:** No PR merges without the other developer's review if it touches a shared contract file (Section 2.2).
- **Definition of Done (global):** code + unit tests + lint passing + `docker compose up -d` still succeeds from a clean checkout + task's own Acceptance Criteria met.
- **Daily async stand-up:** yesterday / today / blockers, referencing task IDs.

---

## 3. Sprint Plan Overview

| Sprint | Theme | Primary Assignee | Depends On | Exit Demo |
|---|---|---|---|---|
| **Sprint 1** | Project Initialization, Tooling & Docker Compose Skeleton | BE + FE (parallel) | - | `docker compose up -d` starts `db`, `pgadmin`, empty `backend`, empty `frontend`; all healthy. |
| **Sprint 2** | Backend Core: DB Schema, Migration, Job State Layer | BE | Sprint 1 | `backend` boots, runs migration, `jobs` row exists with `status='idle'`, visible in pgAdmin. |
| **Sprint 3** | Backend API: `POST /process`, `GET /process`, Validation, Image Store, Process Control | BE | Sprint 2 | `curl` upload -> `202` -> poll -> `processing` -> `done` with base64 image + JSON, all driven by Mock ML. |
| **Sprint 4** | Frontend Core UI Shell & App State | FE | Sprint 1 (can start in parallel with Sprint 2/3) | SPA loads at `localhost:3000`, layout renders, typed API client compiles against the contract. |
| **Sprint 5** | Frontend Upload + Polling Workflow | FE | Sprint 4 + Sprint 3 (API live) | User selects an OPG, sees preview, submits, sees a live "processing" spinner that resolves to raw results. |
| **Sprint 6** | Frontend Interactive Canvas Viewer + Detail Panels | FE | Sprint 5 | Full viewer: boxes, masks, PCA axes, FDI labels, pan/zoom, click-to-select tooth, detail panel, findings table. |
| **Sprint 7** | Integration Hardening, Failure Paths, CORS, Docs, Demo | BE + FE | Sprints 3 + 6 | Clean-machine `docker compose up -d`; full happy path and `fail` path demoed end to end. |

**Integration checkpoints** (`INT-1`..`INT-4`) are interleaved and described in Section 12.

---

## 4. Sprint 1 - Project Initialization, Tooling & Docker Compose Skeleton

**Goal:** A clean `git clone` followed by `docker compose up -d` brings up four healthy containers with no application logic yet.

---

### BE-1.1 - Monorepo scaffold & root tooling

- **Assignee:** BE
- **Target Path:** repository root - `docker-compose.yml`, `.env.example`, `.gitignore`, `README.md`
- **Technical Requirements:**
  - Create the top-level layout from `project-structure.md` Section 9: `backend/`, `frontend/` directories with placeholder `README.md` in each. **Do not create `ml-service/`.**
  - `.gitignore` must cover: `node_modules/`, `dist/`, `build/`, `.env`, `*.local`, `coverage/`, `.DS_Store`, `frontend/.vite/`, `shared/` (local dev mount if any).
  - `.env.example` seeded with every key from `project-structure.md` Section 13.4 that Phase 1 needs:
    ```
    # --- Database ---
    POSTGRES_USER=caries
    POSTGRES_PASSWORD=caries_dev_pw
    POSTGRES_DB=status_db
    DATABASE_URL=postgresql://caries:caries_dev_pw@db:5432/status_db

    # --- Backend ---
    BACKEND_PORT=8000
    MAX_IMAGE_MB=25
    MIN_IMAGE_WIDTH=1000
    MIN_IMAGE_HEIGHT=500
    SHARED_DIR=/shared
    POLL_INTERVAL_MS=2000

    # --- ML driver (Phase 1) ---
    ML_DRIVER=mock            # mock | http
    ML_SERVICE_URL=http://ml-service:8001   # unused in Phase 1, kept for Phase 2
    MOCK_ML_DELAY_MS=6000     # simulated inference time
    MOCK_ML_FAILURE_RATE=0    # 0..1, for testing the fail path

    # --- Frontend ---
    FRONTEND_PORT=3000
    FRONTEND_API_BASE_URL=http://localhost:8000
    VITE_POLL_INTERVAL_MS=2000

    # --- pgAdmin ---
    PGADMIN_DEFAULT_EMAIL=dev@example.com
    PGADMIN_DEFAULT_PASSWORD=admin
    ```
  - Root `README.md` skeleton with sections: "Prerequisites", "Quick start (`cp .env.example .env && docker compose up -d`)", "Ports", "Developer notes".
- **Acceptance Criteria (DoD):**
  - [ ] `git clone` shows `backend/`, `frontend/`, `docker-compose.yml`, `.env.example`, `README.md`, `.gitignore`.
  - [ ] `ml-service/` does **not** exist.
  - [ ] `.env.example` contains all keys above; `cp .env.example .env` produces a working config.

---

### BE-1.2 - Backend Fastify + TypeScript bootstrap

- **Assignee:** BE
- **Target Path:** `backend/package.json`, `backend/tsconfig.json`, `backend/src/server.ts`
- **Technical Requirements:**
  - `package.json` dependencies (pin exact versions): `fastify`, `@fastify/multipart`, `@fastify/cors`, `pg`, `zod`. Dev deps: `typescript`, `tsx` (or `ts-node-dev`), `@types/node`, `vitest`, `eslint`, `@typescript-eslint/*`, `prettier`.
  - Scripts: `"dev": "tsx watch src/server.ts"`, `"build": "tsc -p tsconfig.json"`, `"start": "node dist/server.js"`, `"migrate": "node dist/db/migrate.js"` (wired in `BE-1.4`), `"test": "vitest run"`, `"lint": "eslint src"`.
  - `tsconfig.json`: `"target": "ES2022"`, `"module": "NodeNext"`, `"moduleResolution": "NodeNext"`, `"strict": true`, `"outDir": "dist"`, `"rootDir": "src"`, `"esModuleInterop": true`, `"skipLibCheck": true`.
  - `src/server.ts`: create Fastify instance with `logger: true`; register a stub `GET /health` returning `{ "status": "ok" }` (backend liveness, distinct from any ML health); listen on `process.env.BACKEND_PORT ?? 8000`, host `0.0.0.0`. Route registration and startup migration hooks are stubbed with `// TODO BE-1.4 / BE-3.x` comments.
- **Acceptance Criteria (DoD):**
  - [ ] `npm run dev` starts Fastify locally, `curl localhost:8000/health` -> `{"status":"ok"}`.
  - [ ] `npm run build` emits `dist/server.js` with zero TS errors under `strict`.
  - [ ] `npm run lint` passes.

---

### BE-1.3 - Backend Dockerfile

- **Assignee:** BE
- **Target Path:** `backend/Dockerfile`, `backend/.dockerignore`
- **Technical Requirements:**
  - Multi-stage: `builder` stage on `node:20-slim` runs `npm ci` + `npm run build`; `runtime` stage on `node:20-slim` copies `dist/`, `node_modules` (prod only via `npm ci --omit=dev`), `package.json`.
  - Container entrypoint: `sh -c "npm run migrate && node dist/server.js"` (migration script is a no-op placeholder until `BE-2.2`).
  - Non-root: `USER node`.
  - `EXPOSE 8000`.
  - `.dockerignore`: `node_modules`, `dist`, `.env`, `coverage`, `*.md`.
- **Acceptance Criteria (DoD):**
  - [ ] `docker build ./backend` succeeds.
  - [ ] Running the image alone (no DB) starts Fastify and serves `/health` (migration placeholder must not crash without a DB yet - guard with try/catch + log, see `BE-2.2`).

---

### FE-1.4 - Frontend Vite + React 18 + TypeScript bootstrap

- **Assignee:** FE
- **Target Path:** `frontend/package.json`, `frontend/tsconfig.json`, `frontend/vite.config.ts`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/public/`
- **Technical Requirements:**
  - Scaffold with `npm create vite@latest frontend -- --template react-ts`, then pin versions: `react@18`, `react-dom@18`, `vite@5`, `typescript@5`.
  - Dev deps: `vitest`, `@testing-library/react`, `@testing-library/user-event`, `jsdom`, `eslint`, `prettier`, `@types/react`, `@types/react-dom`.
  - `vite.config.ts`:
    - `server.port` and `preview.port` = `Number(process.env.FRONTEND_PORT ?? 3000)`.
    - `preview.host = true` (so the container binds `0.0.0.0`).
    - `preview.strictPort = true`.
    - `define` / use `import.meta.env` for `VITE_FRONTEND_API_BASE_URL` and `VITE_POLL_INTERVAL_MS`.
  - `index.html`: title "Dental Caries Surface Classification", favicon from `public/`.
  - `src/main.tsx`: standard React 18 `createRoot`.
  - `src/App.tsx`: renders a placeholder `<h1>` only (real layout in `FE-4.1`).
  - Scripts: `"dev"`, `"build": "tsc && vite build"`, `"preview": "vite preview"`, `"test": "vitest run"`, `"lint"`.
- **Acceptance Criteria (DoD):**
  - [ ] `npm run dev` serves the placeholder at `localhost:3000`.
  - [ ] `npm run build` produces `frontend/dist/` with no TS errors.
  - [ ] `npm run preview` serves the built SPA on port 3000, bound to `0.0.0.0`.

---

### FE-1.5 - Frontend Dockerfile (`vite preview`, no Nginx)

- **Assignee:** FE
- **Target Path:** `frontend/Dockerfile`, `frontend/.dockerignore`
- **Technical Requirements:**
  - Single or multi-stage on `node:20-slim`. Steps: `npm ci` -> `npm run build` -> `CMD ["npm", "run", "preview"]`.
  - Build-time args / runtime env: `VITE_*` values are baked at build time, so pass `FRONTEND_API_BASE_URL` and poll interval as Docker build args mapped into `VITE_*` **or** document that a rebuild is required on change. Prefer build args: `ARG VITE_FRONTEND_API_BASE_URL` / `ENV`.
  - `EXPOSE 3000`. No Nginx, no `serve`, no reverse proxy (per `project-structure.md` 11.8).
  - `USER node`.
- **Acceptance Criteria (DoD):**
  - [ ] `docker build ./frontend` succeeds.
  - [ ] `docker run -p 3000:3000 <img>` serves the SPA; `curl localhost:3000` returns the HTML shell.

---

### BE-1.6 - `docker-compose.yml` - four-service skeleton

- **Assignee:** BE (FE reviews the `frontend` block)
- **Target Path:** `docker-compose.yml`
- **Technical Requirements:**
  - Services and settings per `project-structure.md` Sections 13.1-13.3, **minus `ml-service`**:

    | Service | Image / Build | Ports (host:container) | depends_on | Healthcheck |
    |---|---|---|---|---|
    | `db` | `postgres:16` | `127.0.0.1:5432:5432` | - | `pg_isready -U $POSTGRES_USER` interval 5s, retries 10 |
    | `pgadmin` | `dpage/pgadmin4` | `127.0.0.1:5050:80` | `db` | - (optional service) |
    | `backend` | `build: ./backend` | `0.0.0.0:8000:8000` | `db: { condition: service_healthy }` | `wget -qO- localhost:8000/health` |
    | `frontend` | `build: ./frontend` | `0.0.0.0:3000:3000` | `backend` | `wget -qO- localhost:3000` |
  - Volumes: `db_data -> /var/lib/postgresql/data`, `pgadmin_data -> /var/lib/pgadmin`, `shared_data -> /shared` mounted into **`backend` only** in Phase 1 (Phase 2 adds `ml-service` to the same volume).
  - `env_file: .env` on `backend` and `frontend`; explicit `environment:` block on `db` and `pgadmin`.
  - `backend` gets `SHARED_DIR=/shared` and the `shared_data` mount.
  - Bind `db` and `pgadmin` to `127.0.0.1` only (per `project-structure.md` 12.2).
- **Acceptance Criteria (DoD):**
  - [ ] `docker compose config` validates with no warnings.
  - [ ] `docker compose up -d` from a clean checkout brings all four to healthy/running.
  - [ ] `localhost:3000` shows the FE placeholder; `localhost:8000/health` returns ok; `localhost:5050` shows pgAdmin login.
  - [ ] `docker compose down && docker compose up -d` is repeatable; `db_data` persists.

---

### FE-1.7 / BE-1.7 - Shared linting, formatting, CI-lite

- **Assignee:** BE for `backend/`, FE for `frontend/` (identical config)
- **Target Path:** `backend/.eslintrc.cjs`, `backend/.prettierrc`, `frontend/.eslintrc.cjs`, `frontend/.prettierrc`, root `README.md` (dev-notes section)
- **Technical Requirements:**
  - ESLint: `@typescript-eslint/recommended`, `no-floating-promises` error, `no-explicit-any` warn.
  - Prettier: 2-space, single quotes, trailing comma es5, print width 100.
  - Optional root `Makefile` or npm workspace scripts: `make up`, `make down`, `make lint`, `make test`.
- **Acceptance Criteria (DoD):**
  - [ ] `npm run lint` and `npm run test` succeed in both packages (tests may be trivial placeholders at this point).

---

## 5. Sprint 2 - Backend Core: DB Schema, Migration & Job State Layer

**Goal:** The backend owns and provisions the `jobs` table idempotently on startup and exposes a small, tested data-access layer for the job status singleton.

---

### BE-2.1 - PostgreSQL connection pool

- **Assignee:** BE
- **Target Path:** `backend/src/db/pool.ts`
- **Technical Requirements:**
  - Export a singleton `pg.Pool` created from `process.env.DATABASE_URL`.
  - Pool config: `max: 5`, `idleTimeoutMillis: 30000`, `connectionTimeoutMillis: 5000`.
  - Export `async function healthcheckDb(): Promise<boolean>` running `SELECT 1`.
  - Export `async function closePool(): Promise<void>` for graceful shutdown (call on Fastify `onClose`).
  - No credentials logged.
- **Acceptance Criteria (DoD):**
  - [ ] Unit test with a live `db` container: `healthcheckDb()` resolves `true`.
  - [ ] Importing the module does not open a connection until first query (lazy) or opens exactly one pool (documented).

---

### BE-2.2 - Idempotent migration

- **Assignee:** BE
- **Target Path:** `backend/src/db/migrate.ts`
- **Technical Requirements:**
  - `export async function migrate(): Promise<void>` that runs, in one transaction:
    ```sql
    CREATE TABLE IF NOT EXISTS jobs (
      id            SERIAL PRIMARY KEY,
      status        VARCHAR(50)  NOT NULL DEFAULT 'idle',
      fail_message  VARCHAR(255),
      result_path   VARCHAR(255),
      updated_at    TIMESTAMP    NOT NULL DEFAULT NOW()
    );
    ```
  - Then seed the singleton row if the table is empty:
    ```sql
    INSERT INTO jobs (status)
    SELECT 'idle'
    WHERE NOT EXISTS (SELECT 1 FROM jobs);
    ```
  - Add a `CHECK` constraint or documented app-level guard restricting `status` to `('idle','processing','done','fail')` (per `project-structure.md` Section 8). Prefer:
    ```sql
    ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_status_chk;
    ALTER TABLE jobs ADD CONSTRAINT jobs_status_chk
      CHECK (status IN ('idle','processing','done','fail'));
    ```
  - Callable standalone via `npm run migrate` (`if (require.main === module)` / ESM equivalent) **and** imported by `server.ts` at startup **before** `listen()`.
  - Retry wrapper: up to 10 attempts, 2s apart, to tolerate `db` still finishing its healthcheck.
- **Acceptance Criteria (DoD):**
  - [ ] Running `migrate()` twice in a row is a no-op the second time (no error, no duplicate row).
  - [ ] After `docker compose up -d`, pgAdmin shows table `jobs` with exactly one row, `status='idle'`.
  - [ ] Dropping the table and restarting `backend` re-creates and re-seeds it.

---

### BE-2.3 - Job state data-access helpers

- **Assignee:** BE
- **Target Path:** `backend/src/db/jobs.ts`
- **Technical Requirements:**
  - Type:
    ```ts
    export type JobStatus = 'idle' | 'processing' | 'done' | 'fail';
    export interface JobRow {
      id: number;
      status: JobStatus;
      fail_message: string | null;
      result_path: string | null;
      updated_at: string; // ISO
    }
    ```
  - Functions (all operate on the **latest** row = `ORDER BY id DESC LIMIT 1`, per the singleton state-machine model in `project-structure.md` Section 8):
    - `getCurrentJob(): Promise<JobRow>` - never returns null after migration seed.
    - `setProcessing(): Promise<JobRow>` - sets `status='processing'`, clears `fail_message` and `result_path`, `updated_at=NOW()`.
    - `setDone(resultPath: string): Promise<JobRow>` - `status='done'`, `result_path=resultPath`.
    - `setFail(message: string): Promise<JobRow>` - `status='fail'`, `fail_message` truncated to 255 chars.
    - `resetIdle(): Promise<JobRow>` - test/support helper.
  - Every mutation uses a parameterized query and returns the updated row via `RETURNING *`.
- **Acceptance Criteria (DoD):**
  - [ ] Vitest integration suite against the `db` container covers each transition in `project-structure.md` Section 8's diagram: `idle->processing->done`, `processing->fail`, `fail->processing`, `done->processing`.
  - [ ] `fail_message` longer than 255 chars is safely truncated, not rejected.

---

### BE-2.4 - Wire migration + DB shutdown into `server.ts`

- **Assignee:** BE
- **Target Path:** `backend/src/server.ts`
- **Technical Requirements:**
  - On boot: `await migrate()` (with retry) **before** `app.listen()`. If it ultimately fails, log a fatal error and exit non-zero (Compose will restart per policy).
  - Register `app.addHook('onClose', async () => { await closePool(); })`.
  - Add `GET /health` deep variant: `{ "status": "ok", "db": true|false }` using `healthcheckDb()`.
- **Acceptance Criteria (DoD):**
  - [ ] `docker compose up -d` -> `backend` logs show "migration complete" then "server listening on 8000".
  - [ ] `curl localhost:8000/health` -> `{"status":"ok","db":true}`.
  - [ ] `docker compose stop backend` shows a clean shutdown log (pool closed), no unhandled rejection.

---

### BE-2.5 - CORS configuration

- **Assignee:** BE
- **Target Path:** `backend/src/server.ts` (registration), config read from env
- **Technical Requirements:**
  - Register `@fastify/cors` with `origin` restricted to `process.env.FRONTEND_API_BASE_URL`-derived origin, default `http://localhost:3000` (per `project-structure.md` 11.8 / 12.2).
  - Allowed methods: `GET, POST, OPTIONS`. Allowed headers: `Content-Type`. `credentials: false`.
- **Acceptance Criteria (DoD):**
  - [ ] A browser `fetch` from `localhost:3000` succeeds; a request with `Origin: http://evil.example` is rejected by CORS.

---

## 6. Sprint 3 - Backend API: Endpoints, Validation, Image Store, Process Control, Mock ML

**Goal:** The full frontend-facing API from `project-structure.md` Section 7.1 works end to end, with the Mock ML driver producing realistic `processing -> done` and `processing -> fail` outcomes.

---

### BE-3.1 - Authoritative image validation

- **Assignee:** BE
- **Target Path:** `backend/src/lib/validation.ts`
- **Technical Requirements:**
  - `export async function validateOpg(buf: Buffer, mimetype: string): Promise<{ width: number; height: number }>`.
  - Checks (throw typed errors on failure):
    - MIME in `['image/jpeg','image/png']` -> else `UnsupportedMediaTypeError` (maps to HTTP `415`).
    - Byte length `<= MAX_IMAGE_MB * 1024 * 1024` -> else `InvalidImageError` `422`.
    - Decodable image; read intrinsic dimensions using a light dependency (`image-size` package, no full image lib needed) -> else `InvalidImageError` `422`.
    - `width >= MIN_IMAGE_WIDTH` and `height >= MIN_IMAGE_HEIGHT` (env) -> else `InvalidImageError` `422` with message "image resolution too low for OPG analysis".
  - Export the error classes with a `statusCode` and `publicMessage` field.
- **Acceptance Criteria (DoD):**
  - [ ] Unit tests: valid JPEG passes; PDF/GIF -> 415; 1x1 PNG -> 422; oversized buffer -> 422; corrupt bytes -> 422.
  - [ ] Error messages never echo the file name or content.

---

### BE-3.2 - Image store + base64 encoding

- **Assignee:** BE
- **Target Path:** `backend/src/lib/imageStore.ts`
- **Technical Requirements:**
  - `const SHARED_DIR = process.env.SHARED_DIR ?? '/shared'`.
  - `export async function writeInput(buf: Buffer): Promise<string>` - atomically writes to `${SHARED_DIR}/input.jpg` (write to `input.jpg.tmp` then `rename`), overwriting any previous input (per `project-structure.md` 4.7 / 10.3). Returns the absolute path.
  - `export async function readInputAsDataUri(): Promise<string>` - reads `${SHARED_DIR}/input.jpg`, returns `data:image/jpeg;base64,<...>`.
  - `export async function readResultJson(resultPath: string): Promise<unknown>` - reads and `JSON.parse`s the result file from the shared volume.
  - Ensure `SHARED_DIR` exists on startup (`mkdir -p`).
  - Never log file contents; log only byte counts and paths.
- **Acceptance Criteria (DoD):**
  - [ ] Writing twice leaves exactly one `input.jpg` (latest wins).
  - [ ] `readInputAsDataUri()` output is a valid data URI that a browser `<img>` can render.
  - [ ] Concurrent `writeInput` calls do not produce a truncated file (atomic rename verified in a test).

---

### BE-3.3 - ML client abstraction (driver interface)

- **Assignee:** BE
- **Target Path:** `backend/src/services/mlClient.ts`
- **Technical Requirements:**
  - Define the driver interface that Phase 2's HTTP client will also implement:
    ```ts
    export interface MlDriver {
      infer(): Promise<void>;   // fire-and-forget start; resolves once accepted
      cancel(): Promise<void>;  // terminate the running (mock) job if any
      health(): Promise<{ ready: boolean }>;
    }
    ```
  - `export function getMlDriver(): MlDriver` - returns the mock driver when `process.env.ML_DRIVER === 'mock'` (Phase 1 default), otherwise the HTTP driver (Phase 2 stub that throws "not implemented in Phase 1").
  - Keep **all** knowledge of the ML contract in this file (per `project-structure.md` 10.4).
- **Acceptance Criteria (DoD):**
  - [ ] With `ML_DRIVER=mock`, `getMlDriver()` returns the mock; with `ML_DRIVER=http` it returns a stub whose methods throw a clear "Phase 2" error.
  - [ ] `routes/process.ts` and `processControl.ts` import only `getMlDriver`, never the concrete mock.

---

### BE-3.4 - Mock ML driver

- **Assignee:** BE
- **Target Path:** `backend/src/services/mockMl.ts` **(new file - not in `project-structure.md` Section 9; add it under `backend/src/services/`)**
- **Technical Requirements:**
  - Implements `MlDriver`.
  - `infer()`:
    1. Records an internal `AbortController` / timer handle so `cancel()` can stop it (single-flight; only one mock job at a time).
    2. After `MOCK_ML_DELAY_MS` (env, default 6000):
       - With probability `MOCK_ML_FAILURE_RATE` (env, default 0): call `jobs.setFail('Mock ML: simulated failure during detection stage')`.
       - Otherwise: read `${SHARED_DIR}/input.jpg` to get real dimensions (reuse `image-size`), generate a **schema-valid `result.json`** (see `INT-2` fixture shape) with 2-4 synthetic teeth: plausible FDI numbers, bounding boxes inside the image, a simple polygon mask (rectangle or octagon around the bbox), `axes` with a random `rotation_deg`, and 5 `surfaces` each with a random `label`/`probability`. Include `meta.models` = `{ "detector": "mock-det-v0", "classifier": "mock-surf-v0" }` and `meta.timings_ms`.
       - Write it atomically to `${SHARED_DIR}/result.json`, then `jobs.setDone('/shared/result.json')`.
  - `cancel()`: clears the pending timer / aborts; does **not** touch the DB (the caller resets state). Safe to call when nothing is running.
  - `health()`: always `{ ready: true }`.
  - The mock must be deterministic when `MOCK_ML_SEED` is set (optional nice-to-have) so FE tests are stable.
- **Acceptance Criteria (DoD):**
  - [ ] After `infer()`, within `MOCK_ML_DELAY_MS + buffer`, the `jobs` row is `done` and `/shared/result.json` exists and validates against the `INT-2` fixture schema.
  - [ ] Setting `MOCK_ML_FAILURE_RATE=1` makes every run end `fail` with a `fail_message`.
  - [ ] Calling `cancel()` mid-delay prevents the DB from ever reaching `done` for that run.
  - [ ] Generated bboxes and mask polygons are within `[0,width] x [0,height]`.

---

### BE-3.5 - Single-flight process control

- **Assignee:** BE
- **Target Path:** `backend/src/services/processControl.ts`
- **Technical Requirements:**
  - Module-level state: `let active: boolean` (or a small state object) tracking whether an inference is in flight (per `project-structure.md` 10.2 / 6.2).
  - `export async function startNewRun(imageBuffer: Buffer): Promise<void>`:
    1. `await getMlDriver().cancel()` - terminate any running (mock) job.
    2. `await writeInput(imageBuffer)` - overwrite `/shared/input.jpg`.
    3. `await jobs.setProcessing()` - reset the status row.
    4. `await getMlDriver().infer()` - start the new (mock) run.
    5. Set `active = true`; clear it in a `.finally` / when the driver reports completion (the mock can invoke an injected `onSettled` callback, or `processControl` can poll `jobs.getCurrentJob()` — pick the callback approach and document it).
  - Guarantee: even under two near-simultaneous `POST /process` calls, the latest image wins and only one mock job is ever pending. Use a simple in-process mutex (`p-limit(1)` or an awaited promise chain).
  - No queue, no `cancelled` state (per `project-structure.md` 6.2).
- **Acceptance Criteria (DoD):**
  - [ ] Firing `startNewRun(A)` then `startNewRun(B)` 100ms later results in `input.jpg == B` and exactly one `done` with B's dimensions.
  - [ ] The preempted run never writes its `result.json` after being cancelled (verified by content check).
  - [ ] Status row is `processing` immediately after `startNewRun` returns.

---

### BE-3.6 - `POST /process` route

- **Assignee:** BE
- **Target Path:** `backend/src/routes/process.ts`
- **Technical Requirements:**
  - Register `@fastify/multipart` with `limits: { fileSize: MAX_IMAGE_MB * 1024 * 1024, files: 1 }`.
  - Handler:
    1. Read the single `image` field (per `project-structure.md` 7.1). If absent -> `422`.
    2. Buffer the file; run `validateOpg(buf, mimetype)`.
    3. On validation error -> respond with `error.statusCode` (`415` or `422`) and `{ "status": "fail", "fail_message": error.publicMessage }`.
    4. On success -> `await startNewRun(buf)`, then respond `202` with `{ "status": "processing" }`.
  - The handler must return **immediately** after `202` (do not await the mock inference).
  - Structured logs: `{ event: 'process.accepted', bytes, width, height }` - never the file name.
- **Acceptance Criteria (DoD):**
  - [ ] `curl -F image=@valid_opg.jpg localhost:8000/process` -> `202 {"status":"processing"}` in < 500 ms.
  - [ ] `curl -F image=@notes.pdf ...` -> `415`.
  - [ ] `curl -F image=@tiny.png ...` -> `422` with a helpful message.
  - [ ] Request with no `image` field -> `422`.
  - [ ] Uploading a file larger than `MAX_IMAGE_MB` -> `415`/`413` handled gracefully (documented which).

---

### BE-3.7 - `GET /process` route

- **Assignee:** BE
- **Target Path:** `backend/src/routes/process.ts`
- **Technical Requirements:**
  - Handler reads `jobs.getCurrentJob()` and branches exactly per `project-structure.md` Section 7.1:
    - `idle` -> `200 { "status": "idle" }`
    - `processing` -> `200 { "status": "processing" }`
    - `done` -> read `result.json` via `readResultJson(row.result_path)` and the image via `readInputAsDataUri()`; respond:
      ```json
      { "status": "done", "image_base64": "data:image/jpeg;base64,...", "data": { /* result.json contents */ } }
      ```
    - `fail` -> `200 { "status": "fail", "fail_message": row.fail_message }`
  - If `status='done'` but `result.json` is missing/unparseable -> respond `fail` with `fail_message: "result file unavailable"` and log an error (do not 500).
  - Response must be a single self-contained payload (per `project-structure.md` 11.6).
  - No caching headers that would let the browser serve a stale poll (`Cache-Control: no-store`).
- **Acceptance Criteria (DoD):**
  - [ ] Full sequence via `curl`: `POST` -> `GET` returns `processing` -> after `MOCK_ML_DELAY_MS` `GET` returns `done` with a base64 data URI (starts `data:image/jpeg;base64,`) and a `data.teeth` array.
  - [ ] With `MOCK_ML_FAILURE_RATE=1`: `GET` eventually returns `{"status":"fail","fail_message":"..."}`.
  - [ ] Fresh DB (no upload yet) -> `GET` returns `{"status":"idle"}`.
  - [ ] `data` payload validates against the `INT-2` fixture schema.

---

### BE-3.8 - Backend API integration test suite

- **Assignee:** BE
- **Target Path:** `backend/src/routes/__tests__/process.test.ts` **(new)**, `backend/src/fixtures/valid_opg.jpg` **(new)**
- **Technical Requirements:**
  - Use Fastify `app.inject()` + a real `db` container (via `docker compose` test profile or `testcontainers`).
  - Cover: happy path, fail path (force `MOCK_ML_FAILURE_RATE=1`), preemption (two uploads), validation errors (415/422), idle state.
  - Set `MOCK_ML_DELAY_MS=200` in the test env for speed.
- **Acceptance Criteria (DoD):**
  - [ ] `npm test` in `backend/` runs green in CI-lite.
  - [ ] Coverage of `routes/`, `services/`, `lib/`, `db/` >= 80% lines.

---

## 7. Sprint 4 - Frontend Core UI Shell & App State

**Goal:** The SPA renders its final layout, defines all shared types mirroring the API, and ships a typed API client that compiles against the contract. Can be built in parallel with Sprints 2-3 using the `INT-2` fixture.

---

### FE-4.1 - App layout & top-level structure

- **Assignee:** FE
- **Target Path:** `frontend/src/App.tsx`, `frontend/src/main.tsx`
- **Technical Requirements:**
  - `App.tsx` renders a single-column layout:
    - Header: app title + clinic logo (from `public/`), a short "no patient data is stored" notice (per `project-structure.md` 1.1).
    - Main content slot -> renders `<AnalysisView />` (built in `FE-5.1`; stub for now).
    - Footer: build version, port info.
  - No router needed (single view). If added later, keep it to one route `/`.
  - Global styles: a `frontend/src/styles/` folder **(new, under `frontend/src/`)** or CSS Modules per component - pick one and document in `frontend/README.md`.
  - State strategy: local React state + custom hooks only (no Redux/Zustand needed for Phase 1; the app "holds no long-lived state" per `project-structure.md` 4.2). Document this decision.
- **Acceptance Criteria (DoD):**
  - [ ] `localhost:3000` shows header, empty main, footer.
  - [ ] Lighthouse/basic a11y: header is an `<h1>`, landmarks present.
  - [ ] A page refresh fully resets app state (no persistence) - verified.

---

### FE-4.2 - Domain types mirroring the API schema

- **Assignee:** FE
- **Target Path:** `frontend/src/domain/inference.ts`
- **Technical Requirements:**
  - TypeScript types that exactly mirror `project-structure.md` Section 7.1 and the `INT-2` fixture:
    ```ts
    export type ProcessStatus = 'idle' | 'processing' | 'done' | 'fail';

    export interface SurfaceFinding {
      name: 'mesial' | 'distal' | 'occlusal' | 'buccal' | 'lingual';
      label: 'caries' | 'sound';
      probability: number;
    }
    export interface ToothAxes {
      major: [number, number];
      minor: [number, number];
      rotation_deg: number;
    }
    export interface MaskData {
      encoding: 'polygon' | 'rle';
      data: number[][] | number[]; // polygon: [[x,y],...]; rle: flat run lengths
    }
    export interface Tooth {
      id: number;
      fdi: number;
      confidence: number;
      bbox: [number, number, number, number]; // x, y, w, h
      mask: MaskData;
      axes: ToothAxes;
      surfaces: SurfaceFinding[];
    }
    export interface InferenceMeta {
      processed_at: string;
      models: { detector: string; classifier: string };
      timings_ms: Record<string, number>;
    }
    export interface InferenceData {
      meta: InferenceMeta;
      image: { width: number; height: number };
      teeth: Tooth[];
    }

    export type ProcessResponse =
      | { status: 'idle' }
      | { status: 'processing' }
      | { status: 'fail'; fail_message: string }
      | { status: 'done'; image_base64: string; data: InferenceData };
    ```
  - Add a `zod` schema (or hand-written type guard) `parseProcessResponse(json: unknown): ProcessResponse` used by the API client.
- **Acceptance Criteria (DoD):**
  - [ ] Importing the `INT-2` fixture JSON and running `parseProcessResponse` yields no errors.
  - [ ] Types are the **only** representation of the contract on the FE side (no duplicate inline shapes elsewhere).

---

### FE-4.3 - Typed API client

- **Assignee:** FE
- **Target Path:** `frontend/src/api/processClient.ts`
- **Technical Requirements:**
  - Base URL from `import.meta.env.VITE_FRONTEND_API_BASE_URL` (default `http://localhost:8000`).
  - `export async function submitOpg(file: File): Promise<{ status: 'processing' } | { status: 'fail'; fail_message: string }>`:
    - Builds `FormData` with field name `image`.
    - `POST ${base}/process`, `method: 'POST'`, no `Content-Type` header (let the browser set the multipart boundary).
    - Maps `202` -> `{status:'processing'}`; `415`/`422` -> parse body -> `{status:'fail', fail_message}`; network error -> throw `ApiError`.
  - `export async function fetchStatus(signal?: AbortSignal): Promise<ProcessResponse>`:
    - `GET ${base}/process`, `cache: 'no-store'`, pass `signal`.
    - Parse with `parseProcessResponse`.
  - Export `class ApiError extends Error` with `kind: 'network' | 'parse' | 'http'` and optional `status`.
- **Acceptance Criteria (DoD):**
  - [ ] Unit tests with `msw` (Mock Service Worker) or `vi.fn` fetch mock cover: 202, 415, 422, done payload, network failure.
  - [ ] No `any` in the module; all responses flow through `parseProcessResponse`.

---

### FE-4.4 - Client-side validation mirror

- **Assignee:** FE
- **Target Path:** `frontend/src/lib/validation.ts`
- **Technical Requirements:**
  - `export interface OpgConstraints { maxBytes: number; minWidth: number; minHeight: number; acceptedTypes: string[]; }` sourced from `import.meta.env` (`VITE_MAX_IMAGE_MB`, etc.) with sane defaults matching the backend (`BE-3.1`).
  - `export async function preflightOpg(file: File): Promise<{ ok: true; width: number; height: number } | { ok: false; reason: string }>`:
    - Type check against `acceptedTypes` (`image/jpeg`, `image/png`).
    - Size check against `maxBytes`.
    - Decode via `createImageBitmap(file)` to read `width`/`height`; check against `minWidth`/`minHeight`.
  - This is **pre-flight only**; the backend remains authoritative (per `project-structure.md` 4.8). Show friendly messages, but always let the backend have the final say.
- **Acceptance Criteria (DoD):**
  - [ ] Selecting a `.pdf` -> `{ok:false, reason:'Only JPEG or PNG OPG images are accepted'}`.
  - [ ] Selecting a huge file -> friendly size message.
  - [ ] Selecting a tiny image -> resolution message.
  - [ ] Constraint values match the backend defaults (documented cross-check with `BE-3.1`).

---

### FE-4.5 - Mask decoding utilities

- **Assignee:** FE
- **Target Path:** `frontend/src/lib/rle.ts`
- **Technical Requirements:**
  - `export function decodeMask(mask: MaskData, imgW: number, imgH: number): Path2D` OR return an array of polygon rings `number[][]` - pick the representation the renderer (`FE-6.3`) needs and document it.
  - Support `encoding: 'polygon'` (list of `[x,y]` points -> a closed path).
  - Support `encoding: 'rle'` (flat run-length array, row-major) -> either rasterize to an `ImageData` alpha mask or trace to polygons. For Phase 1 the mock emits `polygon` only, so `rle` may be a documented stub that `throw`s "RLE decoding lands in Phase 2" - **but the type and function signature must already exist**.
- **Acceptance Criteria (DoD):**
  - [ ] Given a polygon mask from the `INT-2` fixture, `decodeMask` returns a path/rings that, when filled, visually cover the tooth bbox area.
  - [ ] Unit test: a square polygon decodes to 4 points forming that square.
  - [ ] `rle` path is either implemented or a clearly-labelled `NotImplemented` stub (team decision recorded).

---

## 8. Sprint 5 - Frontend Upload + Polling Workflow

**Goal:** A clinician can select an OPG, see a local preview, submit it, watch a live "processing" state, and land on either raw results or a failure message. Requires the backend API from Sprint 3.

---

### FE-5.1 - Analysis workflow container

- **Assignee:** FE
- **Target Path:** `frontend/src/features/analysis/AnalysisView.tsx`
- **Technical Requirements:**
  - Owns the workflow state machine: `'empty' | 'selected' | 'submitting' | 'processing' | 'done' | 'fail'`.
  - Composition:
    - `'empty' | 'selected'` -> render `<ImageUploader />` (`FE-5.2`).
    - `'submitting' | 'processing'` -> render a progress indicator (indeterminate; note "inference can take 30s or more" per `project-structure.md` 12.3).
    - `'done'` -> render `<CanvasViewer />` + `<ToothDetailPanel />` + `<FindingsTable />` (Sprint 6).
    - `'fail'` -> render an error card with `fail_message` and a "Start over" button.
  - On submit: call `submitOpg(file)`; on `{status:'processing'}` transition to `'processing'` and start polling (`FE-5.3`).
  - "Start over" / selecting a new file while processing: **stop polling immediately** and allow a new upload (per `project-structure.md` 4.2 / 6.2 - a new upload supersedes the running job; the backend handles preemption).
  - Keep the base64 image and `InferenceData` in component state only; clear on "Start over" and on unmount.
- **Acceptance Criteria (DoD):**
  - [ ] Each state renders the correct sub-tree.
  - [ ] Submitting transitions `selected -> submitting -> processing`.
  - [ ] "Start over" from any state returns to `'empty'` and stops polling.
  - [ ] Component unmount aborts in-flight requests (no "setState on unmounted" warnings).

---

### FE-5.2 - Image uploader component

- **Assignee:** FE
- **Target Path:** `frontend/src/components/ImageUploader.tsx`
- **Technical Requirements:**
  - Props:
    ```ts
    interface ImageUploaderProps {
      onValidFile: (file: File, meta: { width: number; height: number }) => void;
      onClear: () => void;
      disabled?: boolean;
    }
    ```
  - Features: click-to-browse `<input type="file" accept="image/jpeg,image/png">` **and** drag-and-drop zone.
  - On file drop/select: run `preflightOpg` (`FE-4.4`). On `ok:false` show the reason inline and do not call `onValidFile`. On `ok:true` show a local `<img>` preview (via `URL.createObjectURL`, revoked on clear/unmount) and a "Submit for analysis" affordance, then call `onValidFile`.
  - Accessibility: the drop zone is keyboard-focusable and announces state via `aria-live`.
- **Acceptance Criteria (DoD):**
  - [ ] Drag-drop and click-browse both work.
  - [ ] Invalid files show a clear inline error and no preview.
  - [ ] Valid file shows a preview and enables submit.
  - [ ] `objectURL` is revoked (no memory leak - verified in a test with a spy).
  - [ ] `disabled` hides/disables all inputs.

---

### FE-5.3 - Polling hook

- **Assignee:** FE
- **Target Path:** `frontend/src/features/analysis/usePolling.ts`
- **Technical Requirements:**
  - Signature:
    ```ts
    function usePolling(opts: {
      enabled: boolean;
      intervalMs?: number; // default import.meta.env.VITE_POLL_INTERVAL_MS ?? 2000
      onResult: (r: ProcessResponse) => void;
      onError: (e: ApiError) => void;
    }): { lastPolledAt: number | null; stop: () => void };
    ```
  - Behavior (per `project-structure.md` 4.2 / 6.1 step 7):
    - When `enabled`, call `fetchStatus` every `intervalMs` using `setTimeout` recursion (not `setInterval`, to avoid overlap).
    - Stop automatically when a response has `status === 'done'` or `status === 'fail'`.
    - Each request carries an `AbortController`; cancel the pending request on `stop()` / unmount / `enabled` flipping to false.
    - Transient network errors: call `onError` but keep polling (with a small backoff cap) unless the caller stops it.
  - Never poll when `enabled` is false.
- **Acceptance Criteria (DoD):**
  - [ ] With fake timers: polls at the configured interval; stops on `done`/`fail`.
  - [ ] `stop()` cancels the in-flight fetch and schedules no more.
  - [ ] No overlapping requests even if the server is slow (next poll waits for the previous to settle).
  - [ ] Unmount cleans up timers and controllers.

---

### FE-5.4 - View-model mapping types

- **Assignee:** FE
- **Target Path:** `frontend/src/features/analysis/analysisTypes.ts`
- **Technical Requirements:**
  - Types that adapt `InferenceData` into what the viewer/panels consume, e.g. a `ToothViewModel` with a precomputed display label (`FDI 36`), a caries summary (`2 / 5 surfaces`), and a stable color key for overlays.
  - `export function toViewModel(data: InferenceData): { image: {width:number;height:number}; teeth: ToothViewModel[]; summary: {...} }`.
  - Pure functions, fully unit-tested; no React imports here.
- **Acceptance Criteria (DoD):**
  - [ ] Fixture -> `toViewModel` yields the expected labels, counts, and per-tooth caries summary.
  - [ ] 100% branch coverage on this module (it is pure and small).

---

## 9. Sprint 6 - Frontend Interactive Canvas Viewer + Detail Panels

**Goal:** Deliver the sole diagnostic output of the app (`project-structure.md` Section 10.1): a layered, interactive HTML5 Canvas viewer with pan/zoom, hit-testing, toggleable layers, plus the per-tooth panel and findings table.

---

### FE-6.1 - Canvas viewer container

- **Assignee:** FE
- **Target Path:** `frontend/src/components/CanvasViewer/CanvasViewer.tsx`
- **Technical Requirements:**
  - Props:
    ```ts
    interface CanvasViewerProps {
      imageBase64: string;          // data URI from GET /process
      data: InferenceData;
      selectedToothId: number | null;
      onSelectTooth: (id: number | null) => void;
    }
    ```
  - Renders a single `<canvas>` sized to its container (ResizeObserver -> update backing store with `devicePixelRatio`).
  - Loads `imageBase64` into an `HTMLImageElement` once; stores natural dimensions.
  - Owns view transform state `{ scale, offsetX, offsetY }`:
    - Wheel -> zoom toward cursor (clamp scale `0.1`..`10`).
    - Pointer drag -> pan.
    - "Fit" and "1:1" buttons.
  - Delegates the actual drawing to `useCanvasRenderer` (`FE-6.3`); delegates layer visibility to `layerState` (`FE-6.4`).
  - Pointer click (no drag) -> hit-test via renderer -> `onSelectTooth(idOrNull)`.
- **Acceptance Criteria (DoD):**
  - [ ] Image renders crisply on HiDPI (no blur).
  - [ ] Wheel zooms toward the cursor; drag pans; Fit/1:1 work.
  - [ ] Resizing the window keeps the view stable and sharp.
  - [ ] Clicking a tooth selects it; clicking empty space deselects.

---

### FE-6.2 - Overlay primitives & styling

- **Assignee:** FE
- **Target Path:** `frontend/src/components/CanvasViewer/overlays.ts`
- **Technical Requirements:**
  - Pure drawing helpers operating on a `CanvasRenderingContext2D` **in image-space** (the renderer applies the world transform):
    - `drawBoundingBox(ctx, bbox, opts)` - stroke, optional dashed for selected.
    - `drawMask(ctx, path, opts)` - semi-transparent fill (`globalAlpha ~0.3`), solid thin outline.
    - `drawPcaAxes(ctx, center, axes, opts)` - major axis line + minor axis line + a small rotation arc/label.
    - `drawFdiLabel(ctx, anchor, fdi, opts)` - text with a readable background chip; keep font size constant in **screen** space (counter-scale by `1/scale`).
    - `drawSelectionHalo(ctx, bbox)` - emphasis for the selected tooth.
  - A shared palette: caries surfaces vs sound surfaces get distinct, colorblind-safe hues; per-tooth stable color from `analysisTypes` color key.
  - Constants for stroke widths expressed in screen pixels, divided by `scale` at call time.
- **Acceptance Criteria (DoD):**
  - [ ] Each helper has a unit test rendering to an offscreen canvas and asserting pixels at known coordinates.
  - [ ] Labels and stroke widths stay visually constant while zooming (verified manually + a scale-math test).
  - [ ] Colors meet WCAG AA contrast against a typical grey OPG background.

---

### FE-6.3 - Canvas draw loop / renderer hook

- **Assignee:** FE
- **Target Path:** `frontend/src/components/CanvasViewer/useCanvasRenderer.ts`
- **Technical Requirements:**
  - `function useCanvasRenderer(params: { canvas: HTMLCanvasElement | null; image: HTMLImageElement | null; data: InferenceData; transform: ViewTransform; layers: LayerState; selectedToothId: number | null; }): { hitTest: (screenX: number, screenY: number) => number | null; requestRedraw: () => void }`.
  - Draw order (per `project-structure.md` 10.1): (1) base image, (2) bounding boxes, (3) segmentation masks (decoded via `FE-4.5`), (4) PCA axes, (5) FDI labels. Selected tooth drawn last with a halo.
  - Uses `requestAnimationFrame`; coalesces multiple `requestRedraw` calls into one frame.
  - Applies the world transform (`translate` + `scale`) once, then calls `overlays.*` in image-space.
  - `hitTest`: converts screen coords to image coords, tests against each tooth's mask path (`ctx.isPointInPath`) first, falling back to bbox; returns the topmost tooth id or `null`.
  - Respects `LayerState` toggles - a hidden layer is skipped entirely.
  - The viewer performs **no analysis** - geometry comes only from `data` (per `project-structure.md` 10.1 "Determinism").
- **Acceptance Criteria (DoD):**
  - [ ] All four overlay types render in the correct z-order over the `INT-2` fixture.
  - [ ] Toggling any layer off removes exactly that layer and nothing else.
  - [ ] `hitTest` returns the correct tooth for points inside masks and `null` outside.
  - [ ] Rapid pan/zoom stays smooth (one draw per frame; profiled, no layout thrash).

---

### FE-6.4 - Layer visibility state

- **Assignee:** FE
- **Target Path:** `frontend/src/components/CanvasViewer/layerState.ts`
- **Technical Requirements:**
  - ```ts
    export interface LayerState { boxes: boolean; masks: boolean; axes: boolean; labels: boolean; }
    export const DEFAULT_LAYERS: LayerState = { boxes: true, masks: true, axes: false, labels: true };
    export function useLayerState(): { layers: LayerState; toggle: (k: keyof LayerState) => void; setAll: (v: boolean) => void };
    ```
  - Independent toggles (per `project-structure.md` 10.1 "Layer control"). Render a small control bar in `CanvasViewer` bound to this hook.
- **Acceptance Criteria (DoD):**
  - [ ] Each toggle flips exactly one layer.
  - [ ] Defaults match `DEFAULT_LAYERS`.
  - [ ] State is component-local and resets on "Start over".

---

### FE-6.5 - Tooth detail panel

- **Assignee:** FE
- **Target Path:** `frontend/src/components/ToothDetailPanel.tsx`
- **Technical Requirements:**
  - Props:
    ```ts
    interface ToothDetailPanelProps {
      tooth: ToothViewModel | null; // null -> "select a tooth" empty state
      onClose: () => void;
    }
    ```
  - Shows: FDI number, detection confidence (as %), PCA `rotation_deg`, and a table of all 5 surfaces with `label` and `probability` (bar or badge). Caries surfaces visually emphasized.
  - Updates reactively when `selectedToothId` changes in `AnalysisView`.
- **Acceptance Criteria (DoD):**
  - [ ] Selecting a tooth on the canvas populates the panel.
  - [ ] Empty state shown when nothing is selected.
  - [ ] All five FDI surface names always listed, even if the API omits some (show "n/a").

---

### FE-6.6 - Findings table

- **Assignee:** FE
- **Target Path:** `frontend/src/components/FindingsTable.tsx`
- **Technical Requirements:**
  - Props: `{ teeth: ToothViewModel[]; selectedToothId: number | null; onSelectTooth: (id: number) => void; }`.
  - One row per detected tooth: FDI, confidence, count of caries surfaces, list of caries surface names.
  - Row click selects the tooth (syncs with canvas + detail panel). Selected row highlighted.
  - Sortable by FDI and by caries-surface count. Default sort: caries count desc, then FDI asc.
- **Acceptance Criteria (DoD):**
  - [ ] Table lists every tooth from `data.teeth`.
  - [ ] Clicking a row selects that tooth everywhere.
  - [ ] Sorting works; default sort as specified.
  - [ ] Selection state stays in sync across canvas, table, and panel (single source of truth in `AnalysisView`).

---

## 10. Sprint 7 - Integration Hardening, Failure Paths, Docs & Demo

**Goal:** A clean machine runs `docker compose up -d` and both the happy path and the failure path work end to end through the browser, with the Mock ML driver.

---

### BE-7.1 - Failure-path & edge-case hardening (Backend)

- **Assignee:** BE
- **Target Path:** `backend/src/routes/process.ts`, `backend/src/services/processControl.ts`, `backend/src/lib/imageStore.ts`
- **Technical Requirements:**
  - `GET /process` returns `done` but `result.json` missing/corrupt -> respond `fail` "result file unavailable", log error, do not 500.
  - `POST /process` while a mock job is mid-delay -> preemption verified; old timer cancelled; no stale `result.json`.
  - DB connection lost mid-request -> `503` with `{ "status": "fail", "fail_message": "service temporarily unavailable" }`; Fastify stays up.
  - Add a global error handler that never leaks stack traces or file paths to the client (per `project-structure.md` 12.1 logging rules).
  - Graceful shutdown on `SIGTERM`: stop accepting, drain, close pool.
- **Acceptance Criteria (DoD):**
  - [ ] Each scenario above has an integration test and passes.
  - [ ] `docker compose kill db` then a `GET /process` returns a clean `503`, and recovery works once `db` is back.

---

### FE-7.2 - Failure-path & resilience hardening (Frontend)

- **Assignee:** FE
- **Target Path:** `frontend/src/features/analysis/AnalysisView.tsx`, `frontend/src/features/analysis/usePolling.ts`, `frontend/src/api/processClient.ts`
- **Technical Requirements:**
  - `fail` status -> error card with `fail_message` + "Start over".
  - Backend unreachable during polling -> non-blocking inline "reconnecting..." banner; keep retrying with capped backoff; recover automatically when the backend returns.
  - `submitOpg` network error -> toast/inline error, stay on the upload screen with the file still selected.
  - Very large base64 image -> ensure the canvas still loads (no main-thread freeze beyond image decode; use `img.decode()`).
  - Guard against a `done` payload with zero teeth -> viewer shows the image with an "no teeth detected" note; findings table shows an empty state.
- **Acceptance Criteria (DoD):**
  - [ ] Stopping the backend mid-poll shows "reconnecting"; restarting it resumes without a page reload.
  - [ ] Forcing the mock to fail shows the error card and "Start over" works.
  - [ ] Empty-teeth payload handled gracefully.

---

### INT-3 - End-to-end walkthrough (joint)

- **Assignee:** BE + FE (pair)
- **Target Path:** `README.md` ("Demo script" section), `zz-claude/` (optional screen recording note)
- **Technical Requirements:**
  - From a clean checkout: `cp .env.example .env && docker compose up -d`.
  - Browser at `localhost:3000`: upload a valid OPG -> see preview -> submit -> see "processing" -> after `MOCK_ML_DELAY_MS` see the Canvas viewer with synthetic teeth -> toggle layers -> click a tooth -> see the detail panel and table selection sync -> "Start over".
  - Repeat with `MOCK_ML_FAILURE_RATE=1` -> see the failure card.
  - Verify in pgAdmin (`localhost:5050`) that the single `jobs` row moved `idle -> processing -> done/fail`.
- **Acceptance Criteria (DoD):**
  - [ ] The full script runs on a machine that has never built the project before.
  - [ ] No `ml-service` container is required or referenced.
  - [ ] `README.md` "Demo script" reproduces the walkthrough step by step.

---

### BE-7.3 / FE-7.3 - Documentation finalization

- **Assignee:** BE (backend + compose sections), FE (frontend section)
- **Target Path:** `README.md`, `backend/README.md`, `frontend/README.md`
- **Technical Requirements:**
  - Root `README.md`: prerequisites, quick start, ports table (`3000`, `8000`, `5050`), env var reference (link to `.env.example`), "Phase 1 vs Phase 2" note (ML deferred, Mock ML driver), troubleshooting.
  - `backend/README.md`: architecture of `src/` (routes, services, db, lib), how the Mock ML driver works and how to switch `ML_DRIVER`, how to run tests.
  - `frontend/README.md`: component map, state strategy decision, how `VITE_*` build args work, how to run against a fixture without the backend.
- **Acceptance Criteria (DoD):**
  - [ ] A new developer can go from clone to running demo using only the READMEs.
  - [ ] Every env var in `.env.example` is explained somewhere.

---

### INT-4 - Phase 2 readiness checklist (joint, documented only)

- **Assignee:** BE + FE
- **Target Path:** `zz-claude/task-pm-phase1.md` (this section stays) + a short `zz-claude/phase2-preconditions.md` **(new)**
- **Technical Requirements:** Record what Phase 2 must do to swap the Mock ML for the real FastAPI service with no FE/route/DB changes:
  - Add `ml-service/` per `project-structure.md` Section 9.
  - Add `ml-service` to `docker-compose.yml` + attach it to `shared_data`.
  - Implement the HTTP `MlDriver` in `services/mlClient.ts` (`POST /infer`, `POST /cancel`, `GET /health`).
  - Flip `ML_DRIVER=http`.
  - Confirm the real `result.json` matches the `INT-2` fixture schema (the contract both sides already code against).
- **Acceptance Criteria (DoD):**
  - [ ] `phase2-preconditions.md` exists and lists the five items with owners.

---

## 11. Consolidated Task Checklist

### Backend (BE)

- [ ] BE-1.1 Monorepo scaffold & root tooling - *root: `docker-compose.yml`, `.env.example`, `.gitignore`, `README.md`*
- [ ] BE-1.2 Fastify + TS bootstrap - *`backend/package.json`, `backend/tsconfig.json`, `backend/src/server.ts`*
- [ ] BE-1.3 Backend Dockerfile - *`backend/Dockerfile`, `backend/.dockerignore`*
- [ ] BE-1.6 `docker-compose.yml` four-service skeleton - *`docker-compose.yml`*
- [ ] BE-1.7 Backend lint/format config - *`backend/.eslintrc.cjs`, `backend/.prettierrc`*
- [ ] BE-2.1 PostgreSQL pool - *`backend/src/db/pool.ts`*
- [ ] BE-2.2 Idempotent migration - *`backend/src/db/migrate.ts`*
- [ ] BE-2.3 Job state helpers - *`backend/src/db/jobs.ts`*
- [ ] BE-2.4 Wire migration + shutdown into server - *`backend/src/server.ts`*
- [ ] BE-2.5 CORS configuration - *`backend/src/server.ts`*
- [ ] BE-3.1 Authoritative image validation - *`backend/src/lib/validation.ts`*
- [ ] BE-3.2 Image store + base64 - *`backend/src/lib/imageStore.ts`*
- [ ] BE-3.3 ML client abstraction - *`backend/src/services/mlClient.ts`*
- [ ] BE-3.4 Mock ML driver - *`backend/src/services/mockMl.ts` (new)*
- [ ] BE-3.5 Single-flight process control - *`backend/src/services/processControl.ts`*
- [ ] BE-3.6 `POST /process` route - *`backend/src/routes/process.ts`*
- [ ] BE-3.7 `GET /process` route - *`backend/src/routes/process.ts`*
- [ ] BE-3.8 Backend API integration tests - *`backend/src/routes/__tests__/process.test.ts` (new), `backend/src/fixtures/` (new)*
- [ ] BE-7.1 Backend failure-path hardening - *`backend/src/routes/`, `backend/src/services/`, `backend/src/lib/`*
- [ ] BE-7.3 Backend + compose docs - *`README.md`, `backend/README.md`*

### Frontend (FE)

- [ ] FE-1.4 Vite + React 18 + TS bootstrap - *`frontend/package.json`, `frontend/tsconfig.json`, `frontend/vite.config.ts`, `frontend/index.html`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/public/`*
- [ ] FE-1.5 Frontend Dockerfile (`vite preview`) - *`frontend/Dockerfile`, `frontend/.dockerignore`*
- [ ] FE-1.7 Frontend lint/format config - *`frontend/.eslintrc.cjs`, `frontend/.prettierrc`*
- [ ] FE-4.1 App layout & structure - *`frontend/src/App.tsx`, `frontend/src/main.tsx`*
- [ ] FE-4.2 Domain types mirroring API - *`frontend/src/domain/inference.ts`*
- [ ] FE-4.3 Typed API client - *`frontend/src/api/processClient.ts`*
- [ ] FE-4.4 Client-side validation mirror - *`frontend/src/lib/validation.ts`*
- [ ] FE-4.5 Mask decoding utilities - *`frontend/src/lib/rle.ts`*
- [ ] FE-5.1 Analysis workflow container - *`frontend/src/features/analysis/AnalysisView.tsx`*
- [ ] FE-5.2 Image uploader component - *`frontend/src/components/ImageUploader.tsx`*
- [ ] FE-5.3 Polling hook - *`frontend/src/features/analysis/usePolling.ts`*
- [ ] FE-5.4 View-model mapping - *`frontend/src/features/analysis/analysisTypes.ts`*
- [ ] FE-6.1 Canvas viewer container - *`frontend/src/components/CanvasViewer/CanvasViewer.tsx`*
- [ ] FE-6.2 Overlay primitives & styling - *`frontend/src/components/CanvasViewer/overlays.ts`*
- [ ] FE-6.3 Canvas draw loop / renderer - *`frontend/src/components/CanvasViewer/useCanvasRenderer.ts`*
- [ ] FE-6.4 Layer visibility state - *`frontend/src/components/CanvasViewer/layerState.ts`*
- [ ] FE-6.5 Tooth detail panel - *`frontend/src/components/ToothDetailPanel.tsx`*
- [ ] FE-6.6 Findings table - *`frontend/src/components/FindingsTable.tsx`*
- [ ] FE-7.2 Frontend failure-path & resilience - *`frontend/src/features/analysis/`, `frontend/src/api/`*
- [ ] FE-7.3 Frontend docs - *`frontend/README.md`*

### Joint / Integration

- [ ] INT-1 API contract sign-off (Section 12)
- [ ] INT-2 `result.json` fixture (Section 12)
- [ ] INT-3 End-to-end walkthrough - *`README.md`*
- [ ] INT-4 Phase 2 readiness checklist - *`zz-claude/phase2-preconditions.md` (new)*

---

## 12. Integration Points - When FE and BE Must Collaborate

### INT-1 - API Contract Sign-off (before Sprint 3 and Sprint 4 code)

- **When:** End of Sprint 1 / start of Sprint 2.
- **Who:** BE + FE, 30 minutes.
- **What to agree and freeze:**
  - Exact request shape of `POST /process`: `multipart/form-data`, single field **`image`**, accepted MIME types, size ceiling.
  - Exact status codes: `202` accepted, `415` unsupported media, `422` invalid image.
  - Exact response bodies for `GET /process` for all four states (`idle`, `processing`, `done`, `fail`) - copied verbatim from `project-structure.md` Section 7.1.
  - The `image_base64` field is a **full data URI** (`data:image/jpeg;base64,...`), not raw base64.
  - Polling cadence: ~2s, from `POLL_INTERVAL_MS` / `VITE_POLL_INTERVAL_MS`.
- **Output:** A short "API Contract v1 - FROZEN" note appended to `README.md`; `backend/src/routes/process.ts` schema comments and `frontend/src/domain/inference.ts` must match it exactly.
- **Change control:** Any later change requires a joint sync and a paired PR touching both `process.ts` and `inference.ts`.

### INT-2 - `result.json` Fixture as the Shared Source of Truth (before Sprint 3 BE-3.4 and Sprint 4 FE-4.2)

- **When:** Start of Sprint 2 (so FE can build against it while BE builds the API).
- **Who:** BE authors, FE reviews; 30 minutes.
- **What:** Create a canonical example result file that both sides import:
  - **Path (BE):** `backend/src/fixtures/result.sample.json` **(new)**
  - **Path (FE):** `frontend/src/fixtures/result.sample.json` **(new)** - byte-identical copy (or a symlink documented in README).
  - Contents: the full `data` object from `project-structure.md` Section 7.1 with **3 realistic teeth**, each with `bbox`, a `polygon` mask, `axes`, and all 5 `surfaces`.
- **Rules:**
  - `BE-3.4` (Mock ML) must emit files that validate against this shape.
  - `FE-4.2` (`parseProcessResponse`) must parse this file with zero errors in a unit test.
  - The mask `encoding` in Phase 1 is always `"polygon"`. `"rle"` is typed but not emitted.
- **Output:** Fixture committed on both sides; a `zod` schema in `frontend/src/domain/inference.ts` is the executable contract.

### INT-3 - End-to-End Walkthrough (Sprint 7)

- **When:** After BE Sprint 3 and FE Sprint 6 are both merged.
- **Who:** BE + FE pair for half a day.
- **What:** Run the full demo script (Section 10, `INT-3` task), fix the seams (CORS origin mismatch, data-URI prefix, port bindings, `depends_on` ordering), and write the `README.md` "Demo script".

### INT-4 - Phase 2 Handoff (end of Sprint 7)

- **When:** Phase 1 close-out.
- **Who:** BE + FE + Tech Lead.
- **What:** Produce `zz-claude/phase2-preconditions.md` (Section 10, `INT-4` task) so swapping in the real FastAPI ML service is a contained change behind `services/mlClient.ts` with `ML_DRIVER=http`.

### Continuous collaboration touchpoints

| Trigger | Action |
|---|---|
| BE changes any response field | Ping FE same day; update `inference.ts` + fixture in a paired PR. |
| FE needs a new field for the viewer | Raise it as a contract change request; do **not** add it client-side only. |
| Either side changes an env var name | Update `.env.example` + both READMEs + `docker-compose.yml` in the same PR. |
| CORS / port / origin error during INT-3 | Joint debugging; the fix lands in `backend/src/server.ts` (CORS) or `docker-compose.yml` (ports), never a frontend hack. |

---

## 13. Risks & Mitigations (Phase 1)

| Risk | Impact | Mitigation |
|---|---|---|
| FE blocked waiting for the real API | Schedule slip | `INT-2` fixture + `msw` mocks let FE build Sprints 4-6 independently. |
| Mock ML output drifts from the future real ML output | Rework in Phase 2 | Both are validated against the single `INT-2` fixture schema / `zod` contract. |
| Base64 image inflates the `GET /process` payload | Slow polls | Only the `done` response carries the image; `processing`/`idle` are tiny. FE stops polling on `done`. |
| Preemption race in `processControl` | Stale results shown | In-process mutex (`p-limit(1)`); integration test `BE-3.5` covers the double-upload case. |
| `vite preview` env baked at build time | Wrong API URL in the container | Pass `VITE_*` as Docker build args; document the rebuild-on-change rule in `frontend/README.md`. |
| Canvas performance with large OPGs + many masks | Janky viewer | `requestAnimationFrame` coalescing, one draw per frame, `img.decode()` before first paint, mask paths cached. |
| Scope creep toward ML in Phase 1 | Timeline blow-out | This document: ML tasks are explicitly Phase 2; the only ML surface is `services/mlClient.ts` + `mockMl.ts`. |

---

## 14. Phase 1 "Definition of Done" (Release Gate)

- [ ] `git clone` + `cp .env.example .env` + `docker compose up -d` brings up `db`, `pgadmin`, `backend`, `frontend`, all healthy, on a machine that never built the project.
- [ ] No `ml-service` directory, image, or container exists or is required.
- [ ] Browser happy path: upload -> preview -> submit -> processing -> interactive Canvas viewer (boxes, masks, PCA axes, FDI labels, pan/zoom, click-select) -> detail panel + findings table in sync -> "Start over".
- [ ] Browser failure path (`MOCK_ML_FAILURE_RATE=1`): upload -> processing -> failure card -> "Start over".
- [ ] `jobs` table has exactly one row; transitions `idle -> processing -> done|fail` visible in pgAdmin.
- [ ] Backend unit + integration tests green, >= 80% line coverage on `src/`.
- [ ] Frontend unit tests green for `domain/`, `api/`, `lib/`, `features/analysis/`, and `CanvasViewer/` helpers.
- [ ] CORS restricted to the frontend origin; `db`/`pgadmin` bound to `127.0.0.1`.
- [ ] `README.md`, `backend/README.md`, `frontend/README.md`, and `zz-claude/phase2-preconditions.md` complete.
- [ ] "API Contract v1 - FROZEN" recorded; `inference.ts` and the `result.sample.json` fixture match it.

---

## 15. Team Task Delegation (Phase 1)

This section assigns the tasks defined in Sections 4-12 to the two developers on
the Phase 1 team. It refines the generic `FE` / `BE` labels used earlier into
named ownership. Where a task was labelled `FE` or `BE` in the sprint sections,
the person named here is the one who writes, tests, and ships it.

### 15.1 Developers and Areas of Ownership

| Developer | Role | Area of Ownership |
|---|---|---|
| **Naris** | Backend Lead & Core Engineer | Repository infrastructure and tooling; the entire Node.js / Fastify backend; PostgreSQL schema, migration and data-access layer; Docker and Docker Compose; the frontend-facing REST API; the Mock ML driver; **and** the HTML5 Canvas rendering and geometry layer on the frontend (`CanvasViewer` draw loop, overlay primitives, mask decoding, hit-testing, pan/zoom transforms). |
| **Sukollapat** | Frontend Lead | Frontend application architecture and bootstrapping; UI/UX layout and page structure; the standard React component set (image uploader, findings table, tooth detail panel, layer controls); application/workflow state; the typed API client; client-side validation; and the polling integration with the backend. |

### 15.2 Folder and File Ownership Map

| Path (from Section 9) | Owner |
|---|---|
| `docker-compose.yml` | **Naris** (Sukollapat reviews the `frontend` service block) |
| `.env.example`, `.gitignore`, root `README.md` | **Naris** (Sukollapat adds the `VITE_*` / frontend keys and the frontend README section) |
| `backend/**` (all subdirectories: `src/routes/`, `src/db/`, `src/services/`, `src/lib/`, `Dockerfile`, config) | **Naris** |
| `backend/src/services/mockMl.ts` **(new)** | **Naris** |
| `backend/src/fixtures/**` **(new)** | **Naris** (authors); Sukollapat co-signs the `result.sample.json` shape |
| `frontend/Dockerfile`, `frontend/vite.config.ts`, `frontend/package.json`, `frontend/tsconfig.json`, `frontend/index.html` | **Sukollapat** |
| `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/styles/**` | **Sukollapat** |
| `frontend/src/api/processClient.ts` | **Sukollapat** |
| `frontend/src/domain/inference.ts` | **Sukollapat** (contract mirrored with Naris via INT-1 / INT-2) |
| `frontend/src/lib/validation.ts` | **Sukollapat** |
| `frontend/src/lib/rle.ts` | **Naris** (mask geometry feeds his renderer) |
| `frontend/src/features/analysis/AnalysisView.tsx` | **Sukollapat** |
| `frontend/src/features/analysis/usePolling.ts` | **Sukollapat** |
| `frontend/src/features/analysis/analysisTypes.ts` | **Sukollapat** |
| `frontend/src/components/ImageUploader.tsx` | **Sukollapat** |
| `frontend/src/components/ToothDetailPanel.tsx` | **Sukollapat** |
| `frontend/src/components/FindingsTable.tsx` | **Sukollapat** |
| `frontend/src/components/CanvasViewer/CanvasViewer.tsx` | **Naris** |
| `frontend/src/components/CanvasViewer/useCanvasRenderer.ts` | **Naris** |
| `frontend/src/components/CanvasViewer/overlays.ts` | **Naris** |
| `frontend/src/components/CanvasViewer/layerState.ts` | **Sukollapat** (layer-toggle UI and hook; consumed by Naris's renderer) |
| `frontend/README.md`, `frontend/.eslintrc.cjs`, `frontend/.prettierrc` | **Sukollapat** |
| `backend/README.md`, `backend/.eslintrc.cjs`, `backend/.prettierrc` | **Naris** |
| `zz-claude/phase2-preconditions.md` **(new)** | **Naris** (Sukollapat adds the frontend-readiness lines) |

### 15.3 Joint Items (both developers)

| Item | Naris | Sukollapat |
|---|---|---|
| **INT-1** API Contract v1 freeze | Defines and documents `POST /process` + `GET /process` shapes; encodes them in `backend/src/routes/process.ts` schemas | Encodes the identical shapes in `frontend/src/domain/inference.ts` + `zod` guard |
| **INT-2** `result.sample.json` fixture | Authors the canonical fixture; makes `mockMl.ts` emit files that validate against it | Reviews and co-signs the shape; wires the `zod` schema and a parsing unit test |
| **INT-3** End-to-end walkthrough | Runs the stack, fixes API/CORS/compose seams, writes the backend half of the `README.md` demo script | Drives the browser walkthrough, fixes viewer/upload/polling seams, writes the frontend half |
| **INT-4** Phase 2 readiness checklist | Backend/compose/ML-driver preconditions | Frontend contract-stability confirmation |

---

### 15.4 Sprint 1 - Project Initialization, Tooling & Docker Compose Skeleton

| Task ID | Owner | Assignment detail |
|---|---|---|
| **BE-1.1** Monorepo scaffold & root tooling | **Naris** | Creates the top-level tree (`backend/`, `frontend/` placeholders), root `README.md`, `.gitignore`, and the full `.env.example` with every Phase 1 key (DB, backend, `ML_DRIVER`/`MOCK_ML_*`, frontend, pgAdmin). Does **not** create `ml-service/`. |
| **BE-1.2** Fastify + TypeScript bootstrap | **Naris** | `backend/package.json` (fastify, `@fastify/multipart`, `@fastify/cors`, `pg`, `zod` + dev tooling), `backend/tsconfig.json` (strict, NodeNext), `backend/src/server.ts` with a stub `GET /health`. |
| **BE-1.3** Backend Dockerfile | **Naris** | `backend/Dockerfile` (multi-stage `node:20-slim`, `USER node`, entrypoint `migrate` then `node dist/server.js`), `backend/.dockerignore`. |
| **BE-1.6** `docker-compose.yml` four-service skeleton | **Naris** | Authors `db`, `pgadmin`, `backend`, `frontend` services; healthchecks; `127.0.0.1` binds for `db`/`pgadmin`; `db_data`, `pgadmin_data`, `shared_data` volumes; `shared_data` mounted into `backend` only. **No `ml-service`.** |
| **BE-1.7** Backend lint/format config | **Naris** | `backend/.eslintrc.cjs`, `backend/.prettierrc`. |
| **FE-1.4** Vite + React 18 + TypeScript bootstrap | **Sukollapat** | `frontend/package.json` (react 18, vite 5, vitest, testing-library), `frontend/tsconfig.json`, `frontend/vite.config.ts` (`server.port`/`preview.port` = 3000, `preview.host=true`, `strictPort`), `frontend/index.html`, `frontend/src/main.tsx`, placeholder `frontend/src/App.tsx`, `frontend/public/`. |
| **FE-1.5** Frontend Dockerfile (`vite preview`, no Nginx) | **Sukollapat** | `frontend/Dockerfile` (`node:20-slim`, `npm ci` -> `npm run build` -> `CMD npm run preview`, `VITE_*` build args, `USER node`, `EXPOSE 3000`), `frontend/.dockerignore`. |
| **FE-1.7** Frontend lint/format config | **Sukollapat** | `frontend/.eslintrc.cjs`, `frontend/.prettierrc` (identical rules to the backend). |
| **Compose `frontend` block review** | **Sukollapat** | Reviews and signs off on ports, `env_file`, build args, and `depends_on: backend` in `docker-compose.yml`. |
| **Sprint 1 exit demo** | **Both** | Naris confirms `db` + `pgadmin` + `backend` healthy; Sukollapat confirms `localhost:3000` serves the SPA placeholder and `docker compose up -d` is repeatable from a clean checkout. |

---

### 15.5 Sprint 2 - Backend Core: DB Schema, Migration & Job State Layer

*Entirely Naris. Sukollapat runs Sprint 4 (frontend shell) in parallel during this sprint.*

| Task ID | Owner | Assignment detail |
|---|---|---|
| **BE-2.1** PostgreSQL connection pool | **Naris** | `backend/src/db/pool.ts` - singleton `pg.Pool` from `DATABASE_URL`, `healthcheckDb()`, `closePool()`. |
| **BE-2.2** Idempotent migration | **Naris** | `backend/src/db/migrate.ts` - `CREATE TABLE IF NOT EXISTS jobs (...)` per Section 8 DDL, singleton-row seed, `jobs_status_chk` CHECK constraint, boot retry (10 x 2s), callable via `npm run migrate` and from `server.ts`. |
| **BE-2.3** Job state data-access helpers | **Naris** | `backend/src/db/jobs.ts` - `JobRow` / `JobStatus` types, `getCurrentJob`, `setProcessing`, `setDone(resultPath)`, `setFail(message)`, `resetIdle`, all on the latest row with `RETURNING *`. |
| **BE-2.4** Wire migration + shutdown into server | **Naris** | `backend/src/server.ts` - `await migrate()` before `listen()`, `onClose -> closePool()`, deep `GET /health` (`{status, db}`). |
| **BE-2.5** CORS configuration | **Naris** | `backend/src/server.ts` - `@fastify/cors` restricted to the frontend origin (`http://localhost:3000` default), methods `GET, POST, OPTIONS`, `credentials:false`. |
| **Sprint 2 exit demo** | **Naris** | `backend` boots, runs migration, `jobs` singleton row visible in pgAdmin with `status='idle'`. |

---

### 15.6 Sprint 3 - Backend API: Endpoints, Validation, Image Store, Process Control, Mock ML

*Entirely Naris. This is the sprint that unblocks Sukollapat's Sprint 5.*

| Task ID | Owner | Assignment detail |
|---|---|---|
| **BE-3.1** Authoritative image validation | **Naris** | `backend/src/lib/validation.ts` - `validateOpg(buf, mimetype)`, MIME allowlist (`image/jpeg`, `image/png`), size ceiling, dimension floor, typed `UnsupportedMediaTypeError` (415) / `InvalidImageError` (422). |
| **BE-3.2** Image store + base64 encoding | **Naris** | `backend/src/lib/imageStore.ts` - atomic `writeInput` to `/shared/input.jpg`, `readInputAsDataUri()` -> `data:image/jpeg;base64,...`, `readResultJson(path)`, ensure `SHARED_DIR` exists. |
| **BE-3.3** ML client abstraction | **Naris** | `backend/src/services/mlClient.ts` - `MlDriver` interface (`infer` / `cancel` / `health`), `getMlDriver()` switching on `ML_DRIVER` (`mock` in Phase 1, `http` stub throwing "Phase 2"). |
| **BE-3.4** Mock ML driver | **Naris** | `backend/src/services/mockMl.ts` **(new)** - timed `infer` (`MOCK_ML_DELAY_MS`), `MOCK_ML_FAILURE_RATE` fail path, reads real `input.jpg` dimensions, generates a schema-valid `result.json` (2-4 synthetic teeth: FDI, in-bounds bbox, polygon mask, `axes`, 5 `surfaces`), atomic write, `jobs.setDone` / `jobs.setFail`; `cancel()` clears the pending timer. |
| **BE-3.5** Single-flight process control | **Naris** | `backend/src/services/processControl.ts` - `startNewRun(buf)`: `cancel()` -> `writeInput` -> `jobs.setProcessing()` -> `infer()`; in-process `p-limit(1)` mutex; no queue, no `cancelled` state. |
| **BE-3.6** `POST /process` route | **Naris** | `backend/src/routes/process.ts` - `@fastify/multipart` (single `image` field, `fileSize` limit), validate -> `startNewRun` -> `202 {status:'processing'}`; errors `415` / `422` with `{status:'fail', fail_message}`. |
| **BE-3.7** `GET /process` route | **Naris** | `backend/src/routes/process.ts` - branch on `jobs.getCurrentJob()`: `idle` / `processing` / `done` (attach `readResultJson` + `readInputAsDataUri`) / `fail`; `Cache-Control: no-store`; missing result file -> `fail` "result file unavailable", never 500. |
| **BE-3.8** Backend API integration tests | **Naris** | `backend/src/routes/__tests__/process.test.ts` **(new)**, `backend/src/fixtures/valid_opg.jpg` **(new)** - `app.inject()` against a real `db`, covering happy path, forced fail, preemption, 415/422, idle; `MOCK_ML_DELAY_MS=200` in tests; >= 80% line coverage. |
| **INT-1 / INT-2 kickoff** | **Naris drafts, Sukollapat co-signs** | Naris freezes the API shape in the route schemas and authors `backend/src/fixtures/result.sample.json`; Sukollapat mirrors both into `frontend/src/domain/inference.ts` and `frontend/src/fixtures/result.sample.json`. |
| **Sprint 3 exit demo** | **Naris** | `curl` upload -> `202` -> poll `processing` -> `done` with base64 data URI + `data.teeth`; `MOCK_ML_FAILURE_RATE=1` yields `fail`. |

---

### 15.7 Sprint 4 - Frontend Core UI Shell & App State

*Entirely Sukollapat. Runs in parallel with Naris's Sprint 2.*

| Task ID | Owner | Assignment detail |
|---|---|---|
| **FE-4.1** App layout & top-level structure | **Sukollapat** | `frontend/src/App.tsx` + `frontend/src/main.tsx` - header (title, clinic logo from `public/`, "no patient data stored" notice), main content slot rendering `<AnalysisView />`, footer; global styles strategy under `frontend/src/styles/`; documents the "local React state + hooks only" decision. |
| **FE-4.2** Domain types mirroring the API | **Sukollapat** | `frontend/src/domain/inference.ts` - `ProcessStatus`, `SurfaceFinding`, `ToothAxes`, `MaskData`, `Tooth`, `InferenceMeta`, `InferenceData`, `ProcessResponse`; `zod` `parseProcessResponse`. Kept byte-aligned with Naris's route schema (INT-1) and fixture (INT-2). |
| **FE-4.3** Typed API client | **Sukollapat** | `frontend/src/api/processClient.ts` - `submitOpg(file)` (FormData field `image`, maps 202/415/422), `fetchStatus(signal)` (`cache:'no-store'`, `parseProcessResponse`), `class ApiError` with `kind`. |
| **FE-4.4** Client-side validation mirror | **Sukollapat** | `frontend/src/lib/validation.ts` - `OpgConstraints` from `import.meta.env`, `preflightOpg(file)` (type / size / `createImageBitmap` dimension check); values cross-checked against Naris's `BE-3.1` defaults. |
| **FE-4.5** Mask decoding utilities | **Naris** | `frontend/src/lib/rle.ts` - `decodeMask(mask, w, h)` returning the path/rings representation his renderer needs; `polygon` implemented, `rle` a typed stub (`throw "RLE decoding lands in Phase 2"`); unit test that a square polygon decodes to its four corners. |
| **Sprint 4 exit demo** | **Sukollapat** (Naris reviews `rle.ts`) | SPA loads at `localhost:3000` with the final layout; `parseProcessResponse` parses the INT-2 fixture with zero errors. |

---

### 15.8 Sprint 5 - Frontend Upload + Polling Workflow

*Entirely Sukollapat. Depends on Naris's Sprint 3 API being live.*

| Task ID | Owner | Assignment detail |
|---|---|---|
| **FE-5.1** Analysis workflow container | **Sukollapat** | `frontend/src/features/analysis/AnalysisView.tsx` - workflow state machine (`empty` / `selected` / `submitting` / `processing` / `done` / `fail`); composes `<ImageUploader />`, progress indicator, and (in Sprint 6) the viewer + panels; on submit calls `submitOpg` then starts polling; "Start over" stops polling and clears state; aborts in-flight requests on unmount. |
| **FE-5.2** Image uploader component | **Sukollapat** | `frontend/src/components/ImageUploader.tsx` - props `onValidFile(file, meta)`, `onClear()`, `disabled?`; click-to-browse + drag-and-drop; runs `preflightOpg`; `URL.createObjectURL` preview revoked on clear/unmount; `aria-live` status. |
| **FE-5.3** Polling hook | **Sukollapat** | `frontend/src/features/analysis/usePolling.ts` - `{ enabled, intervalMs, onResult, onError }`; recursive `setTimeout` (no overlap), `AbortController` per request, auto-stop on `done`/`fail`, capped backoff on transient errors, full cleanup on unmount / `enabled=false`. |
| **FE-5.4** View-model mapping types | **Sukollapat** | `frontend/src/features/analysis/analysisTypes.ts` - `ToothViewModel` (display label `FDI 36`, caries summary `2 / 5 surfaces`, stable color key), pure `toViewModel(data)`; 100% branch coverage. |
| **Sprint 5 exit demo** | **Sukollapat** | Select an OPG -> preview -> submit -> live "processing" indicator -> resolves to a raw results state (viewer arrives in Sprint 6). |

---

### 15.9 Sprint 6 - Frontend Interactive Canvas Viewer + Detail Panels

*Split: Naris owns the Canvas rendering and geometry; Sukollapat owns the surrounding React components and the layer-control UI.*

| Task ID | Owner | Assignment detail |
|---|---|---|
| **FE-6.1** Canvas viewer container | **Naris** | `frontend/src/components/CanvasViewer/CanvasViewer.tsx` - props `imageBase64`, `data`, `selectedToothId`, `onSelectTooth`; single `<canvas>` sized via `ResizeObserver` + `devicePixelRatio`; loads the base64 image once; owns the `{ scale, offsetX, offsetY }` view transform (wheel-zoom toward cursor clamped `0.1`-`10`, drag-pan, Fit / 1:1); click (no drag) -> `hitTest` -> `onSelectTooth`. |
| **FE-6.2** Overlay primitives & styling | **Naris** | `frontend/src/components/CanvasViewer/overlays.ts` - image-space drawing helpers `drawBoundingBox`, `drawMask` (alpha ~0.3 fill + outline), `drawPcaAxes` (major/minor lines + rotation arc), `drawFdiLabel` (screen-constant font via `1/scale` counter-scale), `drawSelectionHalo`; colorblind-safe palette; stroke widths defined in screen px and divided by `scale`. |
| **FE-6.3** Canvas draw loop / renderer hook | **Naris** | `frontend/src/components/CanvasViewer/useCanvasRenderer.ts` - `rAF`-coalesced draw loop in the fixed order image -> boxes -> masks (decoded via `rle.ts`) -> axes -> FDI labels, selected tooth last with halo; `hitTest(screenX, screenY)` via `isPointInPath` on mask paths then bbox fallback; skips hidden layers; performs no analysis of its own. |
| **FE-6.4** Layer visibility state | **Sukollapat** | `frontend/src/components/CanvasViewer/layerState.ts` - `LayerState { boxes, masks, axes, labels }`, `DEFAULT_LAYERS` (`axes:false`), `useLayerState()` (`toggle`, `setAll`); renders the layer-toggle control bar shown inside `CanvasViewer`. Naris's `useCanvasRenderer` consumes this state read-only. |
| **FE-6.5** Tooth detail panel | **Sukollapat** | `frontend/src/components/ToothDetailPanel.tsx` - props `tooth: ToothViewModel | null`, `onClose()`; shows FDI, confidence %, PCA `rotation_deg`, and a 5-row surface table (label + probability, caries emphasized, "n/a" for missing surfaces); reacts to `selectedToothId`. |
| **FE-6.6** Findings table | **Sukollapat** | `frontend/src/components/FindingsTable.tsx` - props `teeth`, `selectedToothId`, `onSelectTooth`; one row per tooth (FDI, confidence, caries-surface count + names); row click selects everywhere; sortable by FDI and by caries count; default sort caries-count desc then FDI asc. |
| **Selection-state wiring** | **Sukollapat** (in `AnalysisView`) | Single `selectedToothId` source of truth in `AnalysisView.tsx`, passed down to Naris's `CanvasViewer` and to `ToothDetailPanel` / `FindingsTable`; keeps canvas, table, and panel in sync. |
| **Sprint 6 exit demo** | **Both** | Full viewer: boxes, masks, PCA axes, FDI labels, pan/zoom, click-to-select; detail panel and findings table stay in sync. Naris demos rendering/hit-testing; Sukollapat demos the panels, table, and layer toggles. |

---

### 15.10 Sprint 7 - Integration Hardening, Failure Paths, Docs & Demo

| Task ID | Owner | Assignment detail |
|---|---|---|
| **BE-7.1** Backend failure-path & edge-case hardening | **Naris** | `backend/src/routes/process.ts`, `backend/src/services/processControl.ts`, `backend/src/lib/imageStore.ts` - `done` with missing/corrupt `result.json` -> `fail`; preemption cancels the old timer with no stale result; DB loss -> clean `503`; global error handler with no stack-trace / path leakage; `SIGTERM` drain + pool close. |
| **FE-7.2** Frontend failure-path & resilience hardening | **Sukollapat** | `frontend/src/features/analysis/AnalysisView.tsx`, `usePolling.ts`, `frontend/src/api/processClient.ts` - `fail` error card + "Start over"; non-blocking "reconnecting..." banner with capped backoff and auto-recovery; `submitOpg` network error keeps the file selected; `img.decode()` before first paint; zero-teeth `done` payload handled with an empty state. |
| **FE-7.2 (viewer robustness)** | **Naris** | Ensures `CanvasViewer` / `useCanvasRenderer` tolerate a very large base64 image and a `teeth: []` payload (render the image with a "no teeth detected" note) without freezing the main thread. |
| **INT-3** End-to-end walkthrough | **Both** | From a clean checkout: `cp .env.example .env && docker compose up -d`. Naris fixes API / CORS / compose / `depends_on` seams and writes the backend half of the `README.md` demo script; Sukollapat drives the browser walkthrough (upload -> processing -> viewer -> tooth select -> layer toggles -> Start over, plus the forced-fail run) and writes the frontend half. Both verify the `jobs` row transitions in pgAdmin. |
| **BE-7.3** Backend + compose documentation | **Naris** | Root `README.md` (prerequisites, quick start, ports, env reference, Phase 1 vs Phase 2 note, troubleshooting) and `backend/README.md` (`src/` architecture, Mock ML driver + `ML_DRIVER` switch, running tests). |
| **FE-7.3** Frontend documentation | **Sukollapat** | `frontend/README.md` (component map, state strategy decision, `VITE_*` build-arg behavior, running against the fixture without the backend). |
| **INT-4** Phase 2 readiness checklist | **Both** | `zz-claude/phase2-preconditions.md` **(new)** - Naris lists the backend / compose / ML-driver preconditions (add `ml-service/`, attach `shared_data`, implement the HTTP `MlDriver`, flip `ML_DRIVER=http`, confirm real `result.json` matches the fixture); Sukollapat confirms no frontend changes are required for the swap. |
| **Sprint 7 exit demo** | **Both** | Clean-machine `docker compose up -d`; full happy path and `fail` path demoed end to end through the browser, Mock ML only, no `ml-service` container. |

---

### 15.11 Coordination Points Between Naris and Sukollapat

| Interface | Naris side | Sukollapat side | Sync mechanism |
|---|---|---|---|
| Frontend-facing API | `backend/src/routes/process.ts` schemas | `frontend/src/domain/inference.ts` + `zod` guard | INT-1 freeze; paired PRs on any change |
| `result.json` structure | `backend/src/services/mockMl.ts` output + `backend/src/fixtures/result.sample.json` | `frontend/src/fixtures/result.sample.json` + parsing test | INT-2 fixture is the single source of truth |
| Mask geometry | `frontend/src/lib/rle.ts` `decodeMask` output shape | `analysisTypes.toViewModel` / `AnalysisView` pass `data` through unchanged | Naris publishes the return type; Sukollapat does not transform mask data |
| Layer visibility | `useCanvasRenderer` reads `LayerState` | `layerState.ts` `useLayerState` + control bar | `LayerState` interface + `DEFAULT_LAYERS` frozen before Sprint 6 coding |
| Tooth selection | `CanvasViewer` emits `onSelectTooth`, highlights `selectedToothId` | `AnalysisView` owns `selectedToothId`; `FindingsTable` / `ToothDetailPanel` read it | Single state owner in `AnalysisView.tsx` |
| Environment / ports | `docker-compose.yml`, `.env.example`, `backend` env | `frontend` service block, `VITE_*` keys | Any env rename updates compose + both READMEs in one PR |
| CORS origin | `@fastify/cors` origin = frontend origin | Frontend served on `localhost:3000` | Verified together during INT-3 |
