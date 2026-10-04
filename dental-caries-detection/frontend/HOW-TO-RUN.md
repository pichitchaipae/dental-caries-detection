# How to Run the Frontend

React 18 + TypeScript + Vite SPA. For architecture and folder layout, see
[`README.md`](./README.md).

## Prerequisites

- **Node.js 22.12 or newer** (`node -v` to check)
- npm (ships with Node)
- Docker, only if you want to run the container build

## 1. Install dependencies

```bash
cd frontend
npm install
```

## 2. Run the dev server (mock backend, default)

```bash
npm run dev
```

Open <http://localhost:3000>.

By default the app runs against an **in-browser mock backend** (MSW), so you
do not need the backend or ML service running. The header shows a mock/live
status badge. Upload any OPG image (at least 1000×500 px, up to 10 MB). After
about 4 seconds the mock returns synthetic teeth, with one tooth flagged with
occlusal caries.

## 3. Run the dev server against the real backend

Start the backend first (from the repo root, for example
`docker compose up db backend`), then create `frontend/.env.local`:

```bash
VITE_API_MOCKING=disabled
VITE_FRONTEND_API_BASE_URL=http://localhost:8000
```

and run `npm run dev` again. The badge should now show **live**.

> **Note:** `npm run dev` reads `.env*` files from `frontend/`, **not** the repo
> root `.env`. Also note that the variable must be named
> `VITE_FRONTEND_API_BASE_URL` here. The root `.env` uses
> `FRONTEND_API_BASE_URL`, which only Docker Compose maps for you.

## Environment variables

Every variable is optional, because the code has a default for each one. Put
overrides in `frontend/.env.local`, which is git-ignored by Vite convention.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_MOCKING` | `enabled` | `enabled` = MSW mock in dev; anything else = real backend. Ignored in builds (mock never ships). |
| `VITE_FRONTEND_API_BASE_URL` | `http://localhost:8000` | Backend base URL |
| `VITE_POLL_INTERVAL_MS` | `2000` | How often `GET /process` is polled |
| `VITE_MAX_IMAGE_MB` | `10` | Client-side max upload size |
| `VITE_MIN_IMAGE_WIDTH` / `VITE_MIN_IMAGE_HEIGHT` | `1000` / `500` | Client-side min image dimensions |
| `VITE_MOCK_ML_DELAY_MS` | `4000` | Mock processing time |
| `VITE_MOCK_ML_FAILURE_RATE` | `0` | `0`–`1`; set `1` to force every mock run to fail (tests the error UI) |

To change the port, use a shell variable rather than a `.env` file:
`FRONTEND_PORT=5173 npm run dev`.

`VITE_*` values are baked in at build time. After you change them, restart
`npm run dev` or rebuild.

## 4. Production build (local)

```bash
npm run build      # tsc type-check + vite build → dist/
npm run preview    # serve dist/ on http://localhost:3000 (bound to 0.0.0.0)
```

The production build **never** uses the mock, so it needs a real backend at
`VITE_FRONTEND_API_BASE_URL`.

## 5. Run with Docker

**Full stack (recommended).** Run this from the repo root:

```bash
cp .env.example .env    # first time only; then fill in secrets
docker compose up --build
```

The frontend is served at <http://localhost:3000>. Compose passes
`FRONTEND_API_BASE_URL` and `VITE_POLL_INTERVAL_MS` from the root `.env` as
build args. **Rebuild** (`--build`) after you change them.

**Frontend container only:**

```bash
cd frontend
docker build -t caries-frontend \
  --build-arg VITE_FRONTEND_API_BASE_URL=http://localhost:8000 .
docker run --rm -p 3000:3000 caries-frontend
```

## Quality checks

```bash
npm test               # Vitest (unit + component tests)
npm run typecheck      # tsc --noEmit
npm run lint           # ESLint
npm run format:check   # Prettier (use `npm run format` to fix)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Unsupported engine` / Vitest crashes on start | Upgrade to Node ≥ 22.12 |
| `Port 3000 is already in use` | Stop the other process (often the Docker frontend) or use `FRONTEND_PORT=3001 npm run dev` |
| Badge says mock but you wanted live | Set `VITE_API_MOCKING=disabled` in `frontend/.env.local` and restart dev server |
| Upload fails immediately with network error (live mode) | Backend not running on `VITE_FRONTEND_API_BASE_URL`; check `curl http://localhost:8000/health` |
| Mock not intercepting requests | Hard-refresh the page. `public/mockServiceWorker.js` must exist (it is committed) |
| Env change has no effect | Restart `npm run dev`; for Docker, rebuild the image |
