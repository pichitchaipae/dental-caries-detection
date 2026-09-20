# Frontend

React 18 + TypeScript single-page application, built with Vite. Owner: Sukollapat.

Served in its container by `vite preview` (no Nginx, per
[`docs-md/project-structure.md`](../docs-md/project-structure.md) Section
11.8). `VITE_FRONTEND_API_BASE_URL` and `VITE_POLL_INTERVAL_MS` are baked into
the bundle at build time; changing them requires a rebuild.

## Layout (through Sprint 6)

```
src/
├── main.tsx                    # React 18 bootstrap; starts the MSW mock in dev (see Mocking below)
├── App.tsx / App.module.css    # FE-4.1: header / main / footer shell
├── domain/
│   ├── inference.ts            # FE-4.2: types + zod contract (parseProcessResponse)
│   └── __tests__/inference.test.ts
├── api/
│   └── processClient.ts        # FE-4.3: submitOpg / fetchStatus, real fetch calls
├── lib/
│   ├── validation.ts           # FE-4.4: client-side pre-flight mirror (not authoritative)
│   └── rle.ts                  # FE-4.5: polygon mask decode + point-in-polygon hit test
├── fixtures/
│   └── result.sample.json      # INT-2 fixture (see Mocking below)
├── mocks/                      # (new, not in project-structure.md's tree — see Mocking below)
│   ├── handlers.ts             # MSW handlers simulating POST/GET /process
│   ├── resultFactory.ts        # Generates synthetic teeth sized to the real uploaded image
│   └── browser.ts              # setupWorker(...handlers)
├── features/analysis/
│   ├── AnalysisView.tsx        # FE-5.1: workflow state machine (empty→...→done/fail)
│   ├── usePolling.ts           # FE-5.3: polls GET /process every VITE_POLL_INTERVAL_MS
│   └── analysisTypes.ts
├── components/
│   ├── ImageUploader.tsx       # FE-5.2: drag/drop + pre-flight validation
│   ├── CanvasViewer/           # FE-6.x: pan/zoom/hit-test renderer, layer toggles
│   ├── ToothDetailPanel.tsx
│   └── FindingsTable.tsx
└── styles/global.css           # CSS reset + color tokens; components use CSS Modules
```

**Styling decision (FE-4.1):** CSS Modules per component (`*.module.css`), plus
one global stylesheet for resets and color tokens. No CSS-in-JS, no framework —
kept minimal since there's no design system to integrate with yet.

## Mocking the backend (Sprints 4-6)

Naris's real `POST`/`GET /process` routes don't exist yet (Sprint 3, `BE-3.6`/
`BE-3.7`). Per `docs-md/task-pm-phase1.md`'s own risk table ("`INT-2` fixture +
`msw` mocks let FE build Sprints 4-6 independently"), `src/mocks/` uses
[MSW](https://mswjs.io) to intercept the real `fetch` calls **at the network
level** — `processClient.ts` is the same code that will talk to the real
backend later; only `main.tsx`'s `enableMockingIfNeeded()` call goes away.

- Active only when `import.meta.env.DEV` is true (`npm run dev`) **and**
  `VITE_API_MOCKING=enabled` (the default). Never active in `vite build`/
  `preview` — the Docker image always expects a real backend.
- `src/mocks/resultFactory.ts` reads the actual dimensions of whatever image
  you upload and generates synthetic teeth inside its real bounds (mirrors
  `backend/src/services/mockMl.ts`'s documented behavior, BE-3.4, once that
  exists) — one tooth is always flagged with an occlusal caries surface so
  the highlighting UI has something to show.
- `src/fixtures/result.sample.json` is the **INT-2 contract fixture**, used by
  `domain/__tests__/inference.test.ts` to prove `parseProcessResponse` accepts
  the agreed shape. Its `image_base64` is a 1x1 placeholder pixel (schema
  fixture only — not meant to be rendered); the live demo's realistic image
  comes from `resultFactory.ts` instead. When Naris creates
  `backend/src/fixtures/result.sample.json` (INT-2 kickoff), this file should
  be reconciled to match byte-for-byte per the plan.
- Env vars (see `.env.example`): `VITE_API_MOCKING`, `VITE_MOCK_ML_DELAY_MS`,
  `VITE_MOCK_ML_FAILURE_RATE` (set to `1` to force every run to fail, for
  testing the fail-path UI).

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Vite dev server on `FRONTEND_PORT` (default 3000) |
| `npm run build` | Type-check (`tsc`), then `vite build` to `dist/` |
| `npm run preview` | Serve the built `dist/` with `vite preview`, bound to `0.0.0.0` |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Vitest |
| `npm run lint` | ESLint (flat config, matches `backend/eslint.config.js`) |
| `npm run format` / `format:check` | Prettier |

Requires Node.js 22.12 or newer (matches `backend/`, Vitest 5 requirement).

## Docker

Multi-stage on `node:22-slim`. The runtime stage keeps `vite` and
`@vitejs/plugin-react` as regular dependencies (not devDependencies) because
`vite preview` loads `vite.config.ts`, which imports the plugin, at runtime.

```bash
docker build -t caries-frontend .
docker run --rm -p 3000:3000 caries-frontend
```

Via Compose, `VITE_FRONTEND_API_BASE_URL` / `VITE_POLL_INTERVAL_MS` are passed
as build args from the root `.env` (see `docker-compose.yml`).
