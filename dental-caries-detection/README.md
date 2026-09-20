# Dental Caries Surface Classification

Stateless local AI inference tool for dental clinics. Architecture:
[`docs-md/project-structure.md`](docs-md/project-structure.md). Phase 1 plan:
[`docs-md/task-pm-phase1.md`](docs-md/task-pm-phase1.md).

Phase 1 runs four services (`db`, `pgadmin`, `backend`, `frontend`). The ML
service is replaced by an in-process Mock ML driver in the backend; there is no
`ml-service/` yet.

## Prerequisites

- Docker Desktop (Windows / macOS) or Docker Engine + Compose v2 (Linux)
- Node.js 22 LTS, only for running `backend/` or `frontend/` outside Docker

## Quick start

```bash
cp .env.example .env
docker compose up -d
```

If port `5432` is already used by a local PostgreSQL, set `POSTGRES_HOST_PORT`
in `.env` (for example `5433`) before `docker compose up -d`.

## Ports

| Port | Service | Bound to |
|---|---|---|
| `3000` | Frontend (SPA) | all interfaces |
| `8000` | Backend API (`/health`, `/process`) | all interfaces |
| `5050` | pgAdmin (developer only) | `127.0.0.1` |
| `5432` (`POSTGRES_HOST_PORT`) | PostgreSQL | `127.0.0.1` |

## Developer notes

- Backend: see [`backend/README.md`](backend/README.md).
- Frontend: see [`frontend/README.md`](frontend/README.md).
- pgAdmin login uses `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD`; add a
  server with host `db`, port `5432`, and the `POSTGRES_*` credentials.
