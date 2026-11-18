# Frontend — ภาคเรียน 2 Sprint 1 (ปอน) · 2026-11-18

> วันที่/เวลาดำเนินการ: **2026-11-18** · รัน typecheck/lint/test/build/format:check รอบสุดท้ายเมื่อ 21:47 (UTC+7)
> ผู้รับผิดชอบ: ปอน (Frontend) · Branch: `feat/sprint_1_sem2_be`
> อ้างอิง: [งานวันที่ 2026-11-14](sprint_1_sem2_pond_2026-11-14.md) · [API contract v2 draft](../api-contract-v2-draft.md) §2.4 / ML-10 · [Progress Backend ของไผ่](../be-sem2-sprint1-progress-2026-11-14.md)
> สถานะ: แก้โค้ดแล้ว **ยังไม่ได้ commit/push** · ยังไม่ได้ rebuild Docker image ของ frontend (ดูหัวข้อ 5)

ขอบเขตโค้ด: แก้เฉพาะ `dental-caries-detection/frontend/` เท่านั้น — `backend/` และ `ml-service/` ไม่ถูกแก้
(ตรวจด้วย `git diff --stat -- backend ml-service` ได้ 0 ไฟล์)
ไฟล์ที่เปลี่ยนนอก `frontend/` มีแค่ไฟล์ `.env` (ไม่อยู่ใน git) กับเอกสารฉบับนี้ — ดูหัวข้อ 1

---

## 1. สรุปสิ่งที่ทำและที่คุยกันในวันนี้

### 1.1 ทดสอบ End-to-End ด้วย Docker Compose (ยังไม่แก้โค้ด)

1. ตรวจ `.env` พบว่าเป็นฉบับเก่าที่ copy มาก่อนมีส่วน Phase 2: `ML_DRIVER=mock`, **ไม่มี** `ML_WEIGHTS_HOST_DIR`, `VITE_API_MOCKING` และค่า ML อื่น
   ถ้ารันตรงๆ Compose จะ fail เพราะ bind-mount source ของ ml-service ว่าง
2. สำรอง `.env` เดิมไว้ที่ `.env.bak-20261118`, แล้วแก้ `.env`: เปลี่ยน `ML_DRIVER=mock` → `http`, เพิ่ม `VITE_API_MOCKING=disabled`,
   `ML_WEIGHTS_HOST_DIR` (ชี้ `…/dental-caries-detection/ml-service/weights`) และค่า `ML_*` / `CARIES_*` / `SURFACE_*` / `VITE_*` ตาม `.env.example`
   (`.env` ถูก gitignore; `.env.bak-20261118` เป็นไฟล์ untracked — **ห้าม commit**)
3. ตรวจ weights ครบทั้ง 3 ไฟล์ใน `ml-service/weights/`: `Tooth_seg_pano_20250319.pt`, `Tooth_seg_crop_20250424.pth`, `caries_crop_seg_20261028.pt`
4. `docker compose up -d --build` สำเร็จ ทุก service **healthy**

| Service | ผล |
|---|---|
| backend | `localhost:8000/health` → `{"status":"ok"}` (200); migration `jobs table is up to date` |
| ml-service | `localhost:8001/health` → `ready: true`, `caries_enabled: true`, artifact ทั้ง 3 ตัวเป็น `verified` |
| frontend | `localhost:3000` → 200 |
| db / pgadmin | up (db healthy, พอร์ต 5434; pgadmin 5050) |

Logs ตอน startup ไม่พบ error · เปิดทดสอบที่ `http://localhost:3000`
**ยังไม่ได้ทดสอบ flow จริง** (upload ภาพ → inference → mask overlay) — ตรวจเฉพาะ health/startup

### 1.2 คำถามที่ตอบ (ไม่แก้โค้ด)

**รัน Frontend อย่างเดียวเพื่อ demo ได้ไหม** — ได้ ด้วย `npm run dev` ใน `frontend/` (MSW mock ใน `src/mocks/`, เปิดเองอัตโนมัติ: `VITE_API_MOCKING ?? 'enabled'`,
mock เฉพาะตอน `npm run dev` ไม่ใช่ `vite preview`). Vite อ่าน `.env` ในโฟลเดอร์ `frontend/` ไม่ใช่ `.env` ที่ root. ผลเป็นข้อมูลจำลอง ไม่ได้พิสูจน์ว่า BE/ML ถูกต้อง

**คำถามจากอาจารย์ (ร่างคำตอบ)**

| คำถาม | คำตอบ |
|---|---|
| ทำไม FE ต้องถาม BE ทุก 2 วินาที ทำไมไม่รอแล้วค่อยแสดง | Inference บน CPU ใช้เวลานาน ถ้าให้ request เดียวค้างรอ จะชน timeout ของ proxy/เบราว์เซอร์ (nginx ~60 วินาที, Cloudflare 100 วินาที), ถ้ารีเฟรชหรือเน็ตหลุดจะเสียผล, และกิน connection นาน. แบบ job: POST ได้ job id ทันที งานเก็บใน DB แล้ว FE poll สถานะ — เรียบง่ายและเสถียรสำหรับผู้ใช้ไม่มาก. ทางเลือก: SSE/WebSocket (ซับซ้อนกว่า), ปรับ poll เป็น backoff ได้ |
| ทำไมต้องมี progress เอาออกได้ไหม | เอา progress bar ออกได้ แต่ **การเช็กสถานะเอาออกไม่ได้** (ต้องรู้ว่าเสร็จ/fail) ที่ตัดได้คือส่วนแสดงผล แทนด้วย spinner. ข้อดีของการมี: งานช้า ถ้านิ่งผู้ใช้คิดว่าค้างแล้วกดซ้ำ. **ยังไม่ได้เช็กว่า progress เป็น % จริงหรือแค่สถานะรายขั้นตอน** — อย่าอ้างว่าเป็น % จริงจนกว่าจะตรวจ |
| ถ้าจะ host ออนไลน์บน Plesk (THZ host) ต้องทำอะไร | Plesk shared hosting ส่วนใหญ่รัน Docker ไม่ได้ และ ML ต้องใช้ PyTorch + weights ~420 MB + RAM หลาย GB → ต้องมี **VPS ที่ลง Docker ได้** (RAM ≥ 4 GB) ก่อนอื่นให้ถาม THZ ว่าแพ็กเกจเป็น shared หรือ VPS. ต้องมี subdomain ชี้ A record, HTTPS ผ่าน reverse proxy (Caddy/nginx/Traefik + Let's Encrypt), แก้ `CORS_ORIGIN` / `FRONTEND_API_BASE_URL` เป็น https แล้ว **rebuild frontend** (VITE_* ฝังตอน build), ปิดพอร์ต DB/pgAdmin/ML ไม่ให้เปิดออกนอกเครื่อง (ตอนนี้ BE 8000 และ FE 3000 bind `0.0.0.0`), เปลี่ยนรหัสผ่านทั้งหมด, ตั้ง `client_max_body_size` (BE จำกัด upload 10 MB) และ timeout, serve FE เป็น static แทน `vite preview`, backup volume DB. ภาพเอกซเรย์เป็นข้อมูลสุขภาพ → consent, นโยบายเก็บ/ลบรูป (PDPA). ทางเลือก: FE static บน Plesk + BE/ML/DB บน VPS (ต้องตั้ง CORS + HTTPS ที่ API) |

**"Caries lesions" คืออะไร ต่างจากด้านบนยังไง** — *Caries surfaces* = สรุปว่าฟันผุที่ด้านไหน (mesial/distal/occlusal, ด้านละ 1 รายการ); *Caries lesions* = รอยผุแต่ละจุดที่ตรวจเจอ (มี id, ด้าน, probability, mask)
ซี่ที่ผุ 2 จุดด้าน distal ทั้งคู่ = 1 surface แต่ 2 lesions. lesion ใช้วาด mask ทับภาพ และ probability ของ surface คือ confidence ของ lesion ที่ถูกจัดเข้าด้านนั้น (หลายจุดใช้ค่าสูงสุด)
ข้อควรระวัง: probability เป็นคะแนน detector ที่ **ยังไม่ calibrated**, และการจัด lesion เข้าด้านเป็น heuristic ตามตำแหน่ง ไม่ใช่ที่โมเดลเรียนรู้

### 1.3 วิเคราะห์ทำไมแผงรายละเอียดฟันขึ้น "Unspecified" (FDI 15)

ภาพที่ดู: FDI 15 — `1 caries surface`, ตาราง Surface = Occlusal / Caries / 83%, ตาราง Caries lesions = `#1 · Unspecified · 83%`

- **ตัวเลขถูกต้อง/สอดคล้องกัน:** มี 1 lesion → surface probability (83%) = lesion probability (83%) ตามกติกา "ใช้ค่าสูงสุดของ lesion ในด้านนั้น"
- **Unspecified ไม่ใช่บั๊ก** แต่เป็นสถานะ "ต้นทางยังไม่ส่งข้อมูล" ที่ contract กำหนดไว้เอง (v2 draft: ไม่มี surface → ส่ง `null` → FE แสดง "ไม่ระบุ")
- **ค้างที่ ML service ไม่ใช่ FE/BE:** ML คำนวณด้านของ lesion ไว้แล้วใน `surface_heuristic` แต่ `encode_caries_lesions`
  (`ml-service/app/pipeline/postprocess.py`) ไม่ได้ใส่ `surface`/`id` ลง output. `resultAdapter.ts` ของ BE รับ `surface` ได้ (`nullish`) แต่ไม่ได้รับ → ส่ง `null` → FE แสดง Unspecified
  นี่คืองาน **ML-10** (ตาม progress BE ของไผ่: "ขอเจาทำ ML-10 … ตอนนี้ BE รองรับ fallback แล้ว แต่ยังไม่ได้ติดต่อเจาหรือแก้ ML")
  — เป็นการสรุปจากการอ่านโค้ด/เอกสาร **ยังไม่ได้ยิงภาพเข้า ML จริงเพื่อดู JSON ดิบ**
- **ตารางนี้จำเป็นไหม:** ตอนนี้ให้ข้อมูลน้อย (1 lesion + 1 surface ซ้ำกับตารางบน และคอลัมน์ Surface ว่าง) แต่จะมีประโยชน์เมื่อฟันมีหลายจุด และเมื่อ ML-10 เสร็จ
- **ตัดสินใจ:** ซ่อนเฉพาะคอลัมน์ Surface ฝั่ง FE ชั่วคราว (หัวข้อ 2) ไม่แก้ BE/ML และไม่ถือว่าปิดงาน ML-10

---

## 2. สิ่งที่แก้ในโค้ด (เฉพาะ `frontend/`)

**พฤติกรรมใหม่:** ในตาราง "Caries lesions" ของแผงรายละเอียดฟัน — ถ้า **ทุก lesion ของซี่นั้นมี `surface = null`** จะไม่แสดงคอลัมน์ Surface (หัวและเซลล์) เหลือ Lesion / Probability
ถ้า **มี lesion อย่างน้อย 1 จุดที่มี surface** จะแสดงคอลัมน์กลับมาตามปกติ และ lesion ที่ยังเป็น `null` ในซี่เดียวกันแสดง "Unspecified" (ไม่ปล่อยช่องว่างให้เข้าใจผิด)
→ เมื่อ ML-10 ship ส่ง surface มาครบ คอลัมน์จะกลับมาเอง **ไม่ต้องแก้ FE ซ้ำ**

| ไฟล์ | การแก้ |
|---|---|
| `src/components/ToothDetailPanel.tsx` | เพิ่ม `showLesionSurface = tooth.caries.some((lesion) => lesion.surface !== null)` ครอบ `<th>Surface</th>` และ `<td>` ของ surface; เปลี่ยนคอมเมนต์เดิมให้อธิบายเหตุผล. ไม่แก้ schema / view-model / canvas — ตารางด้านบน (Surface/Finding/Probability) ไม่เปลี่ยน |
| `src/components/__tests__/ToothDetailPanel.test.tsx` | แทนเทสต์เดิม "shows Unspecified when surface is null" ด้วย 2 เทสต์: (1) ทุก lesion `surface: null` → ไม่มีหัวคอลัมน์ Surface, ไม่มีคำว่า Unspecified, ยังมีคอลัมน์ Lesion และ 55%; (2) lesion หนึ่งมี `distal` อีกอัน `null` → มีคอลัมน์ Surface และมี "Unspecified" ในช่องที่ขาด |

ไม่ได้ทำ: ซ่อนทั้งตาราง lesion (ยังอยากให้เห็นจำนวนและ probability ราย lesion เมื่อมีหลายจุด) และไม่แตะข้อมูล `surface: null` จาก mock (`resultFactory.ts`) เพราะยังเป็นเคสทดสอบที่ใช้ได้

---

## 3. Tests และผลรัน (2026-11-18 21:47, `frontend/`)

| คำสั่ง | ผล |
|---|---|
| `npm run typecheck` | ผ่าน |
| `npm run lint` | ผ่าน |
| `npm test` | ผ่าน — 9 ไฟล์ / **62 tests** (เดิม 61; เปลี่ยน 1 เทสต์เป็น 2) |
| `npm run build` | ผ่าน |
| `npm run format:check` | ผ่าน |

ยังไม่ได้ทำ mutation check ของ `showLesionSurface` (เช่น เปลี่ยนเงื่อนไขเป็น `true` แล้วดูว่าเทสต์ fail) — เทสต์ใหม่ครอบคลุมทั้งสองทิศทางของเงื่อนไข

---

## 4. สิ่งที่ยังค้าง / ข้อสังเกต

1. **ML-10 ยังเป็นของฝั่ง ML** (ส่ง `surface`/`id` ราย lesion ใน `encode_caries_lesions`) — ต้องประสานกับ "เจา" ตาม progress BE; การซ่อนคอลัมน์ใน FE เป็นแค่การปิดอาการชั่วคราว
2. ถ้าอาจารย์ถามเรื่อง progress: ตรวจโค้ด polling/progress ว่าเป็นสถานะหรือ % จริงก่อนตอบ
3. ถ้าจะ deploy จริง: ต้องถาม THZ host เรื่องแพ็กเกจ/Docker ก่อน (หัวข้อ 1.2)
4. `.env.bak-20261118` และ `.vscode/` เป็นไฟล์ untracked ที่ไม่ควร commit

---

## 5. สถานะ Docker และการ commit

- Container `frontend` ที่รันอยู่ที่ `http://localhost:3000` ยังเป็น **image เดิมก่อนแก้รอบนี้** (ยังขึ้น Unspecified) — ต้อง `docker compose up -d --build frontend` ถึงจะเห็นการเปลี่ยนแปลง
  (หรือดูผลทันทีด้วย `npm run dev` ใน `frontend/`) — **ยังไม่ได้ทำทั้งสองอย่าง และยังไม่ได้เปิดดูใน browser** ตรวจผ่านเทสต์ + build เท่านั้น
- ยังไม่ได้ commit/push — ถ้า commit ให้ stage เฉพาะ `frontend/src/components/ToothDetailPanel.tsx`, `frontend/src/components/__tests__/ToothDetailPanel.test.tsx` และไฟล์นี้
  (ไม่รวม `.env.bak-20261118`, `.vscode/`, `ml-service/weights/*`, `z-claude-md/proposal.md`)
