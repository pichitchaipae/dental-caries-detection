> **วันที่:** 2026-09-13
> **ขอบเขต:** Backend Sprint 1 (BE-1.1, BE-1.2, BE-1.3, BE-1.6, BE-1.7) ตาม [`task-pm-phase1.md`](task-pm-phase1.md)
> **หมายเหตุ:** ไม่ได้แก้ `task-pm-phase1.md` / `project-structure.md` — ไฟล์นี้บอกว่าควรแก้อะไรในสองไฟล์นั้น

# รายงาน Backend Sprint 1 + สิ่งที่ต้องแก้ในเอกสาร

## 1. สถานะงาน

| Task | สถานะ | ผลตรวจ |
|---|---|---|
| BE-1.1 Monorepo scaffold | ✅ เสร็จ | ไฟล์ครบ, ไม่มี `ml-service/`, `cp .env.example .env` แล้ว `docker compose config` ผ่าน |
| BE-1.2 Fastify + TS bootstrap | ✅ เสร็จ | `npm run dev` และ `npm start` → `GET /health` = `{"status":"ok"}`; `build` / `typecheck` ไม่มี error; test 1/1 ผ่าน |
| BE-1.3 Backend Dockerfile | ✅ เสร็จ | `docker build` ผ่าน (365MB); รันเดี่ยวไม่มี DB → `/health` = ok; user `node` (UID 1000); เขียน `/shared` ได้; node เป็น PID 1; `docker stop` ใช้ 1 วินาที exit 0 |
| BE-1.6 `docker-compose.yml` | ✅ เสร็จ (ยกเว้น `frontend`) | `db` + `backend` healthy, `pgadmin` `/login` = 200; `db` → `127.0.0.1:5433`, `pgadmin` → `127.0.0.1:5050`; เขียน `shared_data` ได้; `down` → `up` แล้วข้อมูลใน `db_data` ยังอยู่ |
| BE-1.7 Lint / format | ✅ เสร็จ | `npm run lint` ผ่าน; ทดสอบด้วยไฟล์ probe แล้ว `no-floating-promises` = error, `no-explicit-any` = warn; `prettier --check` ผ่าน |

### DoD

- [x] `docker build ./backend` สำเร็จ
- [x] รัน image เดี่ยว (ไม่มี DB) แล้ว `/health` ตอบได้
- [x] `docker compose config` ไม่มี warning
- [x] `docker compose up -d db pgadmin backend` → `db`, `backend` healthy, `pgadmin` running
- [x] `localhost:8000/health` = ok, `localhost:5050` เปิดหน้า login pgAdmin ได้
- [x] `docker compose down && docker compose up -d` ซ้ำได้, `db_data` คงอยู่ (ทดสอบด้วยตาราง marker แล้วลบทิ้ง)
- [ ] `frontend` healthy / `localhost:3000` — **ต้องรอ FE-1.4 / FE-1.5 (Sukollapat)** เพราะยังไม่มี `frontend/Dockerfile`

> ข้อสังเกตตอนทดสอบ: Docker Desktop เปิด port รอไว้ก่อนที่แอปใน container จะ listen → `curl --retry-connrefused` ได้ empty reply (ไม่ใช่ connection refused) จึงไม่ retry. ใน script ตรวจให้ loop จนได้ body ที่ไม่ว่าง

## 2. ไฟล์ที่สร้าง

ทั้งหมดอยู่ใน `dental-caries-detection/dental-caries-detection/` (ไม่แตะไฟล์เดิมนอกโฟลเดอร์นี้)

```
.gitignore
.env.example
.env                      # สำหรับเครื่องนี้เท่านั้น (gitignored), POSTGRES_HOST_PORT=5433
README.md
docker-compose.yml
frontend/README.md        # placeholder
backend/
├── .dockerignore
├── .prettierrc
├── Dockerfile
├── README.md
├── eslint.config.js
├── package.json
├── package-lock.json
├── tsconfig.json
├── tsconfig.build.json   # (new) build โดยไม่รวม test
└── src/
    ├── app.ts            # (new) buildApp() สำหรับ app.inject()
    ├── server.ts
    ├── db/migrate.ts     # placeholder จนถึง BE-2.2
    └── __tests__/health.test.ts
```

## 3. สิ่งที่ต้องแก้ใน `task-pm-phase1.md` (จากการลงมือทำจริง)

### 3.1 ต้องแก้ (ถ้าทำตามเอกสารเดิม จะใช้งานไม่ได้)

| # | จุดในเอกสาร | เดิม | ควรเป็น | เหตุผล |
|---|---|---|---|---|
| 1 | BE-1.6 healthcheck `backend`, `frontend` | `wget -qO- localhost:8000/health` | `node -e "fetch('http://localhost:8000/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"` | image `node:*-slim` ไม่มี `wget` / `curl` → healthcheck fail ตลอด, `frontend` จะไม่เคย healthy |
| 2 | BE-1.1 `.env.example` + BE-1.6 port `db` | `127.0.0.1:5432:5432` ตายตัว | `127.0.0.1:${POSTGRES_HOST_PORT:-5432}:5432` และเพิ่ม `POSTGRES_HOST_PORT` | เครื่องที่ติดตั้ง PostgreSQL ไว้แล้ว (เครื่องนี้มี service `postgresql-x64-17`) จะ `up` ไม่ได้เพราะ port ชน |
| 3 | BE-1.1 `.env.example` + BE-2.5 | CORS origin มาจาก `FRONTEND_API_BASE_URL` | เพิ่ม `CORS_ORIGIN=http://localhost:3000` แล้วใช้ค่านี้ | `FRONTEND_API_BASE_URL` คือ URL ของ backend (`:8000`) ถ้าใช้ทำ origin จะบล็อก frontend จริง |
| 4 | BE-1.7 / FE-1.7 | `.eslintrc.cjs` | `eslint.config.js` (flat config) | ESLint 10 เลิกรองรับ `.eslintrc` แล้ว; `no-floating-promises` ต้องเปิด `parserOptions.projectService` ด้วย ไม่งั้น rule ไม่ทำงาน — **FE ต้องใช้แบบเดียวกัน** |
| 5 | BE-1.3 Dockerfile | ไม่ได้ระบุ | `RUN mkdir -p /shared && chown node:node /shared` ก่อน `USER node` | named volume ใหม่จะเป็นของ root → backend ที่รันเป็น `node` เขียน `input.jpg` ไม่ได้ (พังตอน Sprint 3) |
| 6 | BE-1.3 entrypoint | `sh -c "npm run migrate && node dist/server.js"` | `sh -c "node dist/db/migrate.js && exec node dist/server.js"` + ดัก `SIGTERM` ใน `server.ts` | ไม่มี `exec` → `sh` เป็น PID 1 ไม่ส่ง SIGTERM ต่อ; node ที่เป็น PID 1 ไม่มี handler ก็ไม่ยอมปิด → `docker compose stop` ต้องรอ 10 วินาทีแล้วโดน kill, BE-2.4 / BE-7.1 (clean shutdown) ไม่ผ่าน |

### 3.2 ควรแก้ (อัปเดตเวอร์ชัน / ความชัดเจน)

| # | จุดในเอกสาร | เดิม | ควรเป็น | เหตุผล |
|---|---|---|---|---|
| 7 | BE-1.3, FE-1.5, `project-structure.md` §2, §9, §13.1 | `node:20-slim` | `node:22-slim` | Node 20 หมดอายุ (EOL) 30 เม.ย. 2026; Vitest 5 ต้องการ Node ≥ 22.12 |
| 8 | BE-1.2 | ไม่ระบุเวอร์ชัน TypeScript | pin `typescript@5.9.3` | `typescript` ล่าสุดคือ 7.x แต่ `typescript-eslint` รองรับแค่ `<6.1` |
| 9 | BE-1.2 scripts | `"build": "tsc -p tsconfig.json"` | `"build": "tsc -p tsconfig.build.json"` (exclude test) + เพิ่ม `typecheck`, `format`, `format:check` | ไม่งั้นไฟล์ test จะถูก compile ลง `dist/` ใน image |
| 10 | BE-1.2 / `project-structure.md` §9 tree | มีแค่ `server.ts` | เพิ่ม `src/app.ts` (`buildApp()`) | test ต้องสร้าง Fastify instance โดยไม่ `listen()` (BE-3.8 ใช้ `app.inject()`) |
| 11 | BE-1.2 | `"migrate" ... (wired in BE-1.4)` | `(wired in BE-2.2)` | พิมพ์ผิด — BE-1.4 ไม่มี, FE-1.4 เป็นงาน frontend |
| 12 | BE-1.1 `.env.example` | comment ท้ายบรรทัด เช่น `ML_DRIVER=mock  # mock \| http` | ย้าย comment ขึ้นบรรทัดของตัวเอง | `docker run --env-file` ไม่ตัด inline comment → ค่ากลายเป็น `mock  # mock \| http` |
| 13 | BE-1.6 | ไม่มี restart policy | `restart: unless-stopped` | BE-2.4 เขียนว่า "Compose will restart per policy" แต่ไม่เคยกำหนด policy |
| 14 | BE-1.6 `frontend` | "pass `VITE_*` as build args" | ระบุ mapping: `VITE_FRONTEND_API_BASE_URL: ${FRONTEND_API_BASE_URL}`, `VITE_POLL_INTERVAL_MS` | Vite อ่านได้เฉพาะตัวแปรที่ขึ้นต้นด้วย `VITE_` และถูก bake ตอน build |
| 15 | Sprint 1 exit demo / BE-1.6 DoD | "`docker compose up -d` brings all four to healthy" | ระบุว่าต้องรอ FE-1.5 เสร็จก่อน; ระหว่างนั้นใช้ `docker compose up -d db pgadmin backend` | `build: ./frontend` จะ fail ถ้ายังไม่มี `frontend/Dockerfile` |
| 16 | BE-1.2 dependencies | ไม่มี `image-size`, `p-limit` | เพิ่มในรายการ (ติดตั้งตอน Sprint 3) | BE-3.1, BE-3.4, BE-3.5 ต้องใช้ |
| 17 | INT-4 / `project-structure.md` §4.4 | ML service "non-root user" | ระบุว่า user ใน `ml-service` ต้องเป็น UID 1000 (หรือกำหนด UID ร่วมกัน) | ต้องเขียน `/shared` volume เดียวกับ backend (`node` = UID 1000) |
| 18 | BE-1.3 DoD vs BE-2.4 | BE-1.3: ไม่มี DB ต้องยังรันได้ / BE-2.4: migrate fail ต้อง exit | ระบุว่า DoD ของ BE-1.3 ใช้ถึงแค่ Sprint 1 | สองข้อขัดกันหลัง Sprint 2 |
| 19 | BE-1.7 | "Optional root `Makefile`" | ตัดออก หรือใช้ npm scripts แทน | ทีมใช้ Windows ซึ่งไม่มี `make` มาให้ — Sprint 1 ไม่ได้ทำ Makefile |
| 20 | Header `Related Documents`, INT-4 | `zz-claude/...` | `docs-md/...` | path เอกสารย้ายแล้ว |

### 3.3 พบตอนอ่านเอกสาร (ยังไม่ถึง Sprint 1 แต่ต้องแก้ก่อน Sprint 2–3)

`project-structure.md` (9 ก.ย.) เปลี่ยน DB เป็น history table แต่ `task-pm-phase1.md` (30 ส.ค.) ยังเป็น singleton row:

| หัวข้อ | `project-structure.md` | `task-pm-phase1.md` | Task ที่กระทบ |
|---|---|---|---|
| รูปแบบ | 1 submission = 1 row ใหม่ | row เดียว update ทับ | BE-2.2, BE-2.3 |
| `idle` | ไม่เก็บ (ตารางว่าง = idle) | seed row `status='idle'` | BE-2.2, Sprint 2 exit demo, INT-3, §14 |
| สถานะ | มี `superseded` | ไม่มี | BE-2.3, BE-3.7 (`superseded` → ตอบ `processing`) |
| กัน concurrent | unique index `one_active_job` | `p-limit(1)` | BE-3.5 |
| ML update | `WHERE id = :job_id AND status='processing'` | update row ล่าสุด | BE-3.3 (`infer(jobId)`), BE-3.4 |
| คอลัมน์ | `VARCHAR(20)`, `TIMESTAMPTZ`, มี `created_at` | `VARCHAR(50)`, `TIMESTAMP` | BE-2.2 |

จุดอื่น:

- `result.json` / `input.jpg` ใช้ชื่อไฟล์เดียว → run เก่าที่ cancel ช้าเขียนทับของใหม่ได้ ควรตั้งชื่อตาม job (`result-42.json`)
- ไฟล์ใหญ่เกิน: BE-3.1 ว่า `422`, BE-3.6 ว่า `415/413` — `@fastify/multipart` จะตอบ `413`
- PNG ถูกเก็บเป็น `input.jpg` และส่งกลับเป็น `data:image/jpeg`
- FE-4.4 ใช้ `VITE_MAX_IMAGE_MB`, `VITE_MIN_IMAGE_WIDTH/HEIGHT` แต่ไม่มีใน `.env.example`
- `project-structure.md` ขัดกันเอง: §10.2 ยัง "reset status row", §7.1 ไม่มี `jobId` ใน `202` แต่ §8.2 มี

## 4. ปัญหาสภาพแวดล้อม (เครื่องนี้)

1. **Port 5432 ถูกใช้** โดย Windows service `postgresql-x64-17` (Automatic) → `.env` ของเครื่องนี้ตั้ง `POSTGRES_HOST_PORT=5433`
2. **Docker Desktop 4.45.0 crash ตอนเปิด** — log `%LOCALAPPDATA%\Docker\log\host\com.docker.backend.exe.log`:
   `initializing Inference manager: listening on unix://...\Docker\run\dockerInference: remove ...\dockerInference: The file cannot be accessed by the system.`
   - 2026-09-13 08:57 (เวลาไทย) มีการกดปุ่ม **"Reset to factory defaults"** ใน dialog error ของ Docker (ไม่ใช่ Claude กด) — ตรวจภายหลังพบว่า container / volume เดิมยังอยู่ (ไม่ได้ถูกล้าง)
   - ทางแก้: ปิด Docker Desktop ให้หมด → ลบไฟล์ socket ที่เสียใน `%LOCALAPPDATA%\Docker\run\` (`dockerInference`, `userAnalyticsOtlpHttp.sock`) → เปิดใหม่; ถ้ายังไม่หาย ปิด **Settings → AI → Enable Docker Model Runner** หรืออัปเดต Docker Desktop
   - **สถานะ:** ผู้ใช้แก้แล้ว, Docker 28.3.3 ใช้งานได้
3. **มี container ของโปรเจกต์อื่นรันอยู่** (compose project `devops`: `postgres-db` → `5432`, `pgadmin` → `8081`) — ไม่ชนกับ stack นี้ (`5433`, `5050`, `8000`)

## 5. คำสั่งที่ใช้ตรวจ (ผ่านแล้ว)

```bash
cd backend
npm install
npm run typecheck
npm run build
npm test
npm run lint
npm run format:check
npm run dev
curl localhost:8000/health
cd ..
docker compose config --quiet
docker build -t caries-backend:sprint1 ./backend
docker run -d --name caries-be-solo -p 18000:8000 caries-backend:sprint1
docker compose up -d db pgadmin backend
docker compose ps
curl localhost:8000/health
curl -o /dev/null -w '%{http_code}' 127.0.0.1:5050/login
docker compose down && docker compose up -d db pgadmin backend
```
