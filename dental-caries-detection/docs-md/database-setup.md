# Database Setup

## Overview
The dental caries detection system uses a PostgreSQL database to track inference jobs. The `jobs` table is automatically created by the backend upon startup (via `backend/src/db/migrate.ts`).

## Connection Details
The PostgreSQL database runs in a Docker container named `db` (Phase 1 & 2). 

You can connect to the database from the host machine using any database client (e.g., DBeaver, TablePlus, or the bundled **pgAdmin**) with the following credentials (default from `.env.example`):
- **Host:** `localhost` (if outside Docker) or `db` (if inside Docker / pgAdmin)
- **Port:** `5432`
- **Database:** `status_db`
- **Username:** `caries`
- **Password:** `caries_dev_pw`

## Using pgAdmin
The `pgadmin` service is included in the `docker-compose.yml` for easy database management.
1. Open a web browser and navigate to **http://localhost:5050**
2. Log in with:
   - **Email:** `dev@example.com`
   - **Password:** `admin`
3. Add a new server connection:
   - Right-click **Servers** -> **Register** -> **Server...**
   - **Name:** `Dental-DB` (or any name)
   - Go to the **Connection** tab:
     - **Host name/address:** `db`
     - **Port:** `5432`
     - **Maintenance database:** `status_db`
     - **Username:** `caries`
     - **Password:** `caries_dev_pw`
   - Click **Save**.

## Jobs Table Structure
The backend container runs `backend/src/db/migrate.ts` before starting the server. The migration is idempotent and also upgrades tables created by the earlier version. The resulting schema is:

```sql
CREATE TABLE jobs (
    id BIGINT PRIMARY KEY,                    -- millisecond timestamp, allocated by the backend
    status VARCHAR(255) NOT NULL DEFAULT 'processing',
    result_path VARCHAR(255),                 -- written by ml-service
    fail_message VARCHAR(255),                -- written by ml-service or backend
    updated_at TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT jobs_status_check CHECK (status IN ('processing', 'done', 'fail', 'superseded'))
);
-- At most one job can be 'processing' at a time.
CREATE UNIQUE INDEX one_active_job ON jobs ((true)) WHERE status = 'processing';
```

| status | Written by | Meaning |
|---|---|---|
| `processing` | backend (INSERT) | job sent to ml-service, not finished |
| `done` | ml-service | result written to `result_path` |
| `fail` | ml-service, or backend if ml-service refused the job | `fail_message` explains why |
| `superseded` | backend | a newer upload replaced this job before it finished |

The latest job is the row with the highest `id`. Rows are never deleted (ml-service treats a missing row as "no database" and would keep a late result file).

> ⚠️ ml-service depends on the column names above and on the guard `WHERE id = :job_id AND status = 'processing'`. Don't rename columns or change their types without updating `ml-service/app/db.py`.

### Process Flow
1. **Frontend** uploads an image via `POST /process`.
2. **Backend**, in one transaction: marks any `processing` job as `superseded`, then `INSERT`s the new job (`status = 'processing'`) with an id above the current max.
3. **Backend** saves the image to `/shared/input-{jobId}.jpg`.
4. **Backend** calls **ML Service** `POST /cancel` (stops the old job) and `POST /infer {jobId}`. If ML refuses, the backend sets the job to `fail`.
5. **Backend** deletes the `/shared` files of the jobs that were just replaced.
6. **ML Service** runs inference, writes `/shared/result-{jobId}.json` and runs `UPDATE jobs SET status = 'done', result_path = ... WHERE id = :job_id AND status = 'processing'`. If the job was superseded meanwhile, the update matches 0 rows and ML deletes its result file.
7. **Frontend** polls `GET /process`. The backend reads the latest row: `processing`/`superseded` → `processing`, `fail` → `fail_message`, `done` → reads the file at `result_path` and returns it with the image. The response has `Cache-Control: no-store`; if the database is unreachable the backend answers 503 and the frontend retries.
