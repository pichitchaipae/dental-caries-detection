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
The backend automatically runs `CREATE TABLE IF NOT EXISTS jobs` on startup. The schema is:

```sql
CREATE TABLE jobs (
    id BIGINT PRIMARY KEY,
    status VARCHAR(255) NOT NULL DEFAULT 'processing',
    result_path VARCHAR(255),
    fail_message VARCHAR(255),
    updated_at TIMESTAMP
);
```

### Process Flow
1. **Frontend** uploads an image via `POST /process`.
2. **Backend** saves the image to `/shared` and generates a timestamp-based `jobId`.
3. **Backend** `INSERT`s a new row into the `jobs` table with `status = 'processing'`.
4. **Backend** forwards the `jobId` to **ML Service**.
5. **ML Service** runs inference (Detectron2/YOLO).
6. Once complete, **ML Service** writes the result to `/shared/result-{jobId}.json` and executes `UPDATE jobs SET status = 'done', result_path = ...`.
7. **Frontend** continually polls `GET /process`. The Backend checks if the `.json` file exists or checks the database for `status = 'fail'`. Once finished, the frontend receives the JSON data.
