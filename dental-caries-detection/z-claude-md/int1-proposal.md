# INT-1 Sync — Proposal Draft (for Naris + PM, not yet agreed)

> **Status: DRAFT.** Nothing in this file is authoritative. It exists so the
> actual INT-1 sync (`task-pm-phase1.md` §12: "BE + FE, 30 minutes") is a
> quick confirm-or-adjust instead of a cold start. Do not treat any line here
> as decided until Naris and/or the PM sign off — and until then, do not copy
> anything here into `project-structure.md`, `task-pm-phase1.md`, or
> `README.md`.

## Agenda item 1 — Resolve the `jobId` contradiction

**The problem** (already independently flagged twice — see
`docs-md/be-sprint1-report.md` line 108 and `z-claude-md/pond-task.md`'s
2026-09-20 15:25 entry): `project-structure.md` contradicts itself.

- §7.1 (line 344-345): `POST /process` → `202 { "status": "processing" }` — no `jobId`.
- §8.2 (line 479-480): "...returns `202 { "status": "processing", "jobId": 42 }`."

This isn't cosmetic. `task-pm-phase1.md` line 1225 (`BE-3.3`) defines the ML
driver interface as `infer(jobId)`, and Section 8's whole schema (history
table, `SERIAL id`, `superseded` status, the `one_active_job` unique index) is
built around jobs being individually addressable by id — which only matters
if something downstream of the `202` response actually needs that id.

**Two ways to resolve it, both internally consistent — pick one:**

| | Option A: Singleton row (Sukollapat's preference, stated 2026-09-20) | Option B: History table (as currently written in §8) |
|---|---|---|
| `jobs` table | One row, updated in place, seeded `status='idle'` | New row per submission, `superseded` on preemption |
| `202` response | `{ "status": "processing" }` — no `jobId`, ever | `{ "status": "processing", "jobId": 42 }` |
| `GET /process` | Reads the single row | Reads `ORDER BY id DESC LIMIT 1` |
| Matches | `task-pm-phase1.md`'s original Aug 30 draft | `project-structure.md`'s current (Sept 9) §8, §11.4 rationale |
| Backend impact | None yet — `BE-2.x`/`BE-3.x` haven't been built | None yet — same |
| Frontend impact | `frontend/src/domain/inference.ts`'s `ProcessResponse` already matches this (no `jobId` field) — zero change needed | Would need a `jobId` field added to the `done`/`processing`... actually) `ProcessResponse` type + zod schema, a paired PR per §2.2 |

**Why I'm not picking one myself:** §11.4 in `project-structure.md` gives real
rationale for the history-table model (each preempted run gets its own
terminal `superseded` state, distinct from `fail`, so preemptions don't
pollute failure metrics) — that's not a mistake to just delete, it's a design
trade-off Naris or the PM may have intended deliberately when they wrote the
newer version of the doc. Sukollapat's preference is the simpler singleton
model since it requires zero frontend changes, but that's a reason to
*propose* it, not to unilaterally declare it.

**If Option A is agreed:** `project-structure.md` §6.2, §8, §8.1, §8.2, §8.3
need rewriting to drop `superseded`/history semantics, and `be-sprint1-report.md`
§3.3's already-noted plan/doc conflict resolves in the same direction.
**If Option B is agreed:** `frontend/src/domain/inference.ts` needs a `jobId`
field added, in a paired PR with `backend/src/routes/process.ts` per the
change-control rule in `task-pm-phase1.md` §2.2.

## Agenda item 2 — Canvas/mask-geometry file ownership

`task-pm-phase1.md` §15 (lines 1159, 1166-1168, 1246, 1271-1273) assigns these
files to **Naris**:
- `frontend/src/lib/rle.ts`
- `frontend/src/components/CanvasViewer/CanvasViewer.tsx`
- `frontend/src/components/CanvasViewer/useCanvasRenderer.ts`
- `frontend/src/components/CanvasViewer/overlays.ts`

As of 2026-09-20, all four were built by Sukollapat (with working pan/zoom,
hit-testing, layer toggles, brightness/contrast — browser-verified against a
real sample image, zero console errors). Two ways this can go, for Naris/PM
to decide:

1. **Reassign to Sukollapat officially** in `task-pm-phase1.md` §15 (what
   was asked for) — reflects reality, but is Naris's/PM's call since it's
   his named ownership being changed, not a unilateral frontend edit.
2. **Naris reviews and takes over maintenance going forward**, treating
   today's work as a working first draft rather than final ownership — also
   reasonable, since the plan gave those files to him for a reason (likely:
   he's expected to also touch the ML-service-side geometry/mask format in
   Phase 2, so keeping the whole geometry pipeline with one owner end-to-end
   might matter more than who happened to write Phase 1's version first).

No frontend action is blocked on this either way — it's a bookkeeping/process
question, not a functional one.

## Agenda item 3 — "API Contract v1 – FROZEN" note (once 1 and 2 are settled)

Once Naris and Sukollapat agree on the above, this is the line to append to
root `README.md` (per `task-pm-phase1.md` line 1062):

```markdown
## API Contract v1 — FROZEN (2026-XX-XX)

`POST /process` and `GET /process` request/response shapes are frozen per
`docs-md/project-structure.md` Section 7.1[, with `jobId` <included in / omitted
from> the `202` response per the INT-1 sync on 2026-XX-XX — see
`z-claude-md/int1-proposal.md` for the resolution history]. Any further change
requires a joint sync and a paired PR touching both
`backend/src/routes/process.ts` and `frontend/src/domain/inference.ts`
(`task-pm-phase1.md` Section 2.2).
```
