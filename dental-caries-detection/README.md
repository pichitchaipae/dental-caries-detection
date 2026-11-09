# Dental Caries Surface Classification

Stateless local AI inference tool for dental clinics. Architecture:
[`docs-md/project-structure.md`](docs-md/project-structure.md). Phase 1 plan:
[`docs-md/task-pm-phase1.md`](docs-md/task-pm-phase1.md).

The current Docker Compose file defines five services: `frontend`, `backend`,
`ml-service`, `db` (PostgreSQL for processing status), and `pgadmin` (a developer
database console). See [the deployment topology](docs-md/project-structure.md#132-compose-topology).

Compose output can show six created/started items: **five containers and one
network** (`dental-caries-detection_default`). The network connects the services;
it is not another container. Check the containers with `docker compose ps`.

The default `ML_DRIVER=mock` in `.env.example` makes the backend simulate
inference. It does not prevent Compose from starting `ml-service`. Set
`ML_DRIVER=http` to use real inference, with the model weights configured below.

## Prerequisites

- Docker Desktop (Windows / macOS) or Docker Engine + Compose v2 (Linux)
- Node.js 22 LTS, only for running `backend/` or `frontend/` outside Docker

## Quick start

```bash
# First setup only; keep your existing .env if you already have one.
cp .env.example .env
```

Before starting, edit `.env` to use the weights directory on your machine. For
the standard checkout layout, this works on macOS, Linux, and Windows:

```dotenv
ML_WEIGHTS_HOST_DIR=./ml-service/weights
```

Ensure that directory contains the required model files (see
[`ml-service/README.md`](ml-service/README.md)). For real inference, also change
the existing `ML_DRIVER` entry to `http`. Then run from `dental-caries-detection/`:

```bash
docker compose config --quiet
docker compose up -d
docker compose ps
```

If port `5432` is already used by a local PostgreSQL, set `POSTGRES_HOST_PORT`
in `.env` (for example `5433`) before `docker compose up -d`.

## Ports

| Port | Service | Bound to |
|---|---|---|
| `3000` | Frontend (SPA) | all interfaces |
| `8000` | Backend API (`/health`, `/process`) | all interfaces |
| `8001` (`ML_SERVICE_PORT`) | ML inference API (`/health`, `/infer`) | `127.0.0.1` |
| `5050` | pgAdmin (developer only) | `127.0.0.1` |
| `5432` (`POSTGRES_HOST_PORT`) | PostgreSQL | `127.0.0.1` |

## Troubleshooting

### `The "ML_WEIGHTS_HOST_DIR" variable is not set. Defaulting to a blank string.`

Compose uses `ML_WEIGHTS_HOST_DIR` as the host directory mounted read-only at
`/weights` inside `ml-service`. This warning means Compose cannot find a value
when reading `docker-compose.yml`, often because `.env` is missing or an older
copy does not include that variable. A blank mount source can prevent the ML
container from being created.

1. Run commands from the `dental-caries-detection/` directory containing
   `docker-compose.yml`. If `.env` does not exist, copy `.env.example` to `.env`;
   otherwise edit the existing file without overwriting your settings.
2. Add or update this line in `.env`:

   ```dotenv
   ML_WEIGHTS_HOST_DIR=./ml-service/weights
   ```

   Relative paths resolve from the Compose project directory. Alternatively,
   use your actual absolute path, for example:

   ```dotenv
   # macOS example: replace this with your own checkout path.
   ML_WEIGHTS_HOST_DIR="/Users/sukollapat/projects/dental-caries-detection/ml-service/weights"
   ```

   ```dotenv
   # Windows example: use forward slashes; adjust to your checkout path.
   ML_WEIGHTS_HOST_DIR="D:/Mahidol University/SP/dental-caries-detection/ml-service/weights"
   ```

   Choose one value. The Windows path currently in `.env.example` must be
   changed when using another machine. An empty directory removes neither the
   need for weights nor model-loading errors: the directory must contain
   `Tooth_seg_pano_20250319.pt`, `Tooth_seg_crop_20250424.pth`, and
   `caries_crop_seg_20261028.pt` for real inference.
3. Validate and start again:

   ```bash
   docker compose config --quiet
   docker compose up -d
   docker compose ps
   docker compose logs --tail=100 ml-service
   ```

   For real inference, set `ML_DRIVER=http` in `.env` and check
   `http://localhost:8001/health`: the production model entries should report
   `verified` and `caries_enabled` should be `true`. A container reporting
   `Started` alone does not prove that its models loaded successfully.

### Running without pgAdmin

pgAdmin is optional; PostgreSQL is required by the backend. To start the four
application/database services without starting pgAdmin:

```bash
docker compose up -d db backend ml-service frontend
```

If pgAdmin is already running, stop it with `docker compose stop pgadmin`.
Running `docker compose up -d` without service names starts all five services
again. Changing `ML_DRIVER` alone does not change the number of containers.

## Developer notes

- Backend: see [`backend/README.md`](backend/README.md).
- Frontend: see [`frontend/README.md`](frontend/README.md).
- pgAdmin login uses `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD`; add a
  server with host `db`, port `5432`, and the `POSTGRES_*` credentials.
