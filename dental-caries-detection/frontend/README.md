# Frontend

React 18 + TypeScript single-page application, built with Vite. Owner: Sukollapat.

Served in its container by `vite preview` (no Nginx, per
[`docs-md/project-structure.md`](../docs-md/project-structure.md) Section
11.8). `VITE_FRONTEND_API_BASE_URL` and `VITE_POLL_INTERVAL_MS` are baked into
the bundle at build time; changing them requires a rebuild.

## Layout (through Sprint 7)

```
src/
├── main.tsx                    # React 18 bootstrap; starts the MSW mock in dev (see Mocking below)
├── App.tsx                     # FE-4.1: header / main / footer shell + mock/live status badge
├── styles/global.css           # Tailwind entry point (@import 'tailwindcss') + @theme color tokens
├── domain/
│   ├── inference.ts            # FE-4.2: types + zod contract (parseProcessResponse)
│   └── __tests__/inference.test.ts
├── api/
│   └── processClient.ts        # FE-4.3: submitOpg / fetchStatus, real fetch calls, ApiError
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
│   ├── AnalysisView.tsx        # FE-5.1/7.2: workflow state machine + submit/poll error handling
│   ├── usePolling.ts           # FE-5.3/7.2: setTimeout-recursion poll loop, capped backoff on failure
│   ├── analysisTypes.ts        # FE-5.4: toViewModel — ToothViewModel (labels, caries summary, colorKey)
│   └── __tests__/
├── components/
│   ├── ImageUploader.tsx       # FE-5.2: drag/drop + pre-flight validation
│   ├── CanvasViewer/           # FE-6.x: pan/zoom/hit-test renderer, layer toggles, brightness/contrast
│   ├── ToothDetailPanel.tsx    # consumes ToothViewModel
│   └── FindingsTable.tsx       # consumes ToothViewModel; empty-state when teeth.length === 0
```

**Styling:** Tailwind CSS v4 (`@tailwindcss/vite` plugin, see `vite.config.ts`),
utility classes directly in JSX. No CSS Modules, no CSS-in-JS — migrated off
CSS Modules when the UI was redesigned against a reference mockup; brand
colors live as Tailwind `@theme` tokens in `styles/global.css` (`brand-*`,
`danger-*`, `success-*`, `warning-*`).

**State strategy (FE-4.1):** local React state + hooks only, no external
store. `AnalysisView` is the single source of truth for workflow phase,
selected tooth, and submit/connectivity error state; a page refresh discards
everything by design (project-structure.md 4.2 — a refresh also abandons any
in-progress run, since polling stops).

**View-model layer (FE-5.4):** components never consume the raw
`InferenceData`/`Tooth` wire types directly. `analysisTypes.toViewModel(data)`
precomputes `displayLabel` ("FDI 36"), `cariesSummary` ("1 / 5 surfaces"),
`hasCaries`, and a stable per-tooth `colorKey` used for the bounding-box
stroke color on canvas and the identity dot in `FindingsTable`/
`ToothDetailPanel` — so a given tooth reads as the same color everywhere.

**Resilience (Sprint 7, FE-7.2):**
- `submitOpg` network failure keeps the file selected and shows an inline
  error with a retry (`AnalysisView`'s `submitError` state).
- Polling backs off exponentially (capped at 8x the base interval) on
  consecutive failures and shows a non-blocking "Reconnecting to server…"
  note without leaving the processing view; resets on the next success
  (`usePolling.ts`, unit-tested with fake timers).
- A `done` payload with zero teeth shows "No teeth detected" in both the
  canvas hint and the Findings Table instead of an empty-looking view.
- `CanvasViewer` calls `img.decode()` before first paint so a large base64
  image doesn't stall the main thread on the first draw.

## Mocking the backend (Sprints 4-7)

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
