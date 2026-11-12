# API Contract v2 (ร่าง) — แสดง mask รอยผุบนเว็บ

> **สถานะ:** DRAFT · **branch:** `feat/sprint_1_sem2` · **วันที่:** 2026-11-12
> **เจ้าของ:** Backend (ไผ่) · **ผู้ใช้หลัก:** Frontend (ปอน) · **แหล่งข้อมูล ML:** เจา
> ต่อยอดจาก [`api-contract-v1.md`](./api-contract-v1.md) (v1.2) — เอกสารนี้ยังไม่ freeze

---

## 0. เป้าหมายของ Sprint 1 (ภาคการศึกษาที่ 2)

1. ส่ง **mask ของรอยผุ** จากแบบจำลอง YOLOv8s-seg (caries 1 คลาส บนภาพตัดรายซี่) ไปถึงหน้าเว็บ
   เพื่อให้ทันตแพทย์เห็นว่ารอยผุอยู่ตรงไหนของฟัน ไม่ใช่แค่ระบายสีทั้งซี่
2. ปรับตามคอมเมนต์จากการนำเสนอ Proposal (Demo):
   - Bounding box ซ้ำหน้าที่กับ mask → **เอาออกจากหน้าจอ** แสดงเฉพาะฟันและจุดที่มีรอยผุ
   - UX/UI ไม่ผ่าน → เปลี่ยนสีกรอบ / สีที่ใช้แสดงผล
   - Confidence ของ YOLO เป็นข้อมูลเกิน → **เหลือแค่ Probability**
   - เพิ่ม mask ให้เห็นว่ารอยผุอยู่ตรงไหน

---

## 1. สถานะปัจจุบัน (v1.2) — ML → Backend → Frontend

ตรวจจากโค้ดใน branch นี้ (แตกจาก `main` ณ 12 พ.ย.):
`ml-service/app/pipeline/{orchestrator,postprocess,surface_heuristic}.py`,
`backend/src/services/resultAdapter.ts`, `frontend/src/domain/inference.ts`,
`frontend/src/features/analysis/analysisTypes.ts`, `frontend/src/components/CanvasViewer/overlays.ts`

| Field | ML ส่ง (raw) | Backend ส่งต่อ (API v1.2) | Frontend ใช้ |
|---|---|---|---|
| `meta.job_id`, `completed_at`, `image_size` | ✅ | ✅ (`processed_at`, `image`) | ✅ |
| `meta.models` | ✅ ชื่อไฟล์โมเดล | ✅ | – |
| `meta.timings_ms` | ❌ | `{}` | – |
| `meta.tooth_count`, `caries_count`, `caries_lesion_count` | ✅ | ❌ ทิ้ง | – |
| `teeth[].fdi` | ✅ | ✅ | ✅ label |
| `teeth[].confidence` (tooth detection) | ✅ | ✅ | ✅ ตาราง + แผงรายละเอียด ← **ต้องเอาออก** |
| `teeth[].bbox` | ✅ | ✅ | ✅ วาดกรอบ, ตำแหน่ง label, hit-test สำรอง ← **เลิกวาดกรอบ** |
| `teeth[].mask` (polygon ฟัน) | ✅ | ✅ | ✅ ระบายสีทั้งซี่ |
| `teeth[].axes` | ✅ | ✅ | ✅ เส้นแกน |
| `teeth[].surfaces[]` (`name`, `probability`, `method`) | ✅ | ✅ | ✅ แผงรายละเอียด |
| `teeth[].caries[].mask` (polygon รอยผุ) | ✅ | ✅ (v1.1) | ❌ parse ผ่านแต่ **ถูกทิ้งใน `toToothViewModel`** ไม่ได้วาด |
| `teeth[].caries[].bbox`, `centroid` | ✅ | ✅ | ❌ |
| `teeth[].caries[].confidence` | ✅ | ✅ | ❌ |
| ด้านของรอยผุแต่ละจุด (lesion → surface) | ❌ คำนวณใน `classify_lesion_surface` แต่ **ไม่ได้ใส่ใน output** | ❌ | ❌ |

**สรุป:** mask ของรอยผุส่งมาถึง Frontend แล้วตั้งแต่ v1.1 สิ่งที่ขาดคือ
(1) Frontend ยังไม่วาด, (2) ไม่รู้ว่ารอยผุแต่ละจุดเป็นด้านไหน จึงผูก lesion กับ `surfaces[]` ไม่ได้,
(3) ยังส่งและแสดง confidence ของ YOLO ที่ไม่ต้องการแล้ว

> หมายเหตุเรื่องค่า: `surfaces[].probability` ปัจจุบัน **คือ** caries detector confidence ของ lesion
> ที่ถูกจัดเข้าด้านนั้น (ใช้ค่าสูงสุดถ้ามีหลาย lesion) ดู v1.2 · ส่วน `teeth[].confidence` คือความมั่นใจว่า
> "ตรงนี้คือฟันซี่นี้" ไม่เกี่ยวกับรอยผุ

---

## 2. สิ่งที่เสนอให้เปลี่ยนใน v2

### 2.1 Frontend-facing (`GET /process` → `data`)

```
teeth[]
├── id, fdi, mask, axes, has_caries        (เหมือนเดิม)
├── bbox                                    คงไว้ใน API (ใช้วาง label / hit-test) แต่ FE ไม่วาดกรอบ
├── confidence                              ❌ ลบออก (tooth detection confidence)
├── surfaces[]  { name, label, probability, method }   (เหมือนเดิม)
└── caries[]    ← required (ไม่มี = [])
    ├── id           number    ใหม่ — unique ภายในฟันซี่นั้น (0..n-1) ใช้เป็น key / selection
    ├── surface      "mesial" | "distal" | "occlusal"   ใหม่ — ด้านที่ lesion นี้ถูกจัดเข้า
    ├── probability  number 0–1                         เปลี่ยนชื่อจาก confidence
    ├── mask         { encoding: "polygon", data: [[x,y], ...] }   (เดิม)
    └── centroid     [x, y]                             (เดิม)
```

- ลบ `caries[].bbox` ออกจาก FE contract (mask ใช้แทนได้; ถ้าต้องการกรอบให้คำนวณจาก polygon)
- `caries[].probability` คือค่าเดียวกับ confidence เดิมของ YOLO แต่ตั้งชื่อตรงกับ `surfaces[].probability`
  เพื่อให้ทั้งเว็บมีตัวเลขชนิดเดียวคือ "Probability" · ยังเป็นคะแนนที่ **ไม่ผ่าน calibration**
- `surfaces[x].probability` = ค่าสูงสุดของ `caries[].probability` ที่มี `surface == x` (กฎเดิม)
- พิกัดทุกตัวยังเป็น pixel ของรูปต้นฉบับ

### 2.2 ML raw (ขอเจา)

| # | ขอเพิ่ม | ที่อยู่ | เหตุผล |
|---|---|---|---|
| ML-10 | ใส่ `surface` (และ `id`) ให้ทุก entry ใน `teeth[].caries[]` | `surface_heuristic.findings_for_lesions` คำนวณด้านของแต่ละ lesion อยู่แล้ว → ส่งผลกลับให้ `postprocess.encode_caries_lesions` | FE ต้องรู้ว่าจุดไหนคือด้านไหน; ถ้าให้ Backend คำนวณเองต้องเขียนกฎ PCA ซ้ำใน TypeScript ซึ่งเสี่ยงไม่ตรงกับ ML |
| ML-6 | ส่ง `meta.timings_ms` | `orchestrator.py` | ใช้วัดเวลาต่อภาพตาม Evaluation Matrix |

### 2.3 Backend (ไผ่)

- [ ] `resultAdapter.ts`: รับ `caries[].surface` / `id` จาก ML, map `confidence → probability`,
      ตัด `teeth[].confidence` และ `caries[].bbox` ออกจาก output, ตั้ง `caries: []` เป็นค่า default
- [ ] ตรวจความสอดคล้อง: lesion ทุกจุดต้องมี `surface` ที่อยู่ใน `surfaces[]` ของฟันซี่เดียวกัน
      (ถ้าไม่ตรง → log + ยังส่งผล ไม่ fail ทั้งงาน)
- [ ] ระหว่างรอ ML-10: ถ้า lesion ไม่มี `surface` ให้ส่ง `surface: null` (FE แสดง "ไม่ระบุ")
- [ ] อัปเดต fixture `backend/src/fixtures/result.sample.json` + raw sample ให้มี caries mask จริง
- [ ] unit test ของ adapter (ครบทุกกรณี: ไม่มี caries, หลาย lesion ด้านเดียวกัน, lesion ไม่มี surface)
- [ ] bump เอกสารเป็น v2.0 เมื่อ freeze และเปิด PR คู่กับ FE ตามกฎใน v1 §1.4

### 2.4 Frontend (ปอน) — สรุปสิ่งที่กระทบ เพื่อวางแผนร่วมกัน

- `domain/inference.ts`: schema ตาม §2.1
- `analysisTypes.ts`: ส่ง `caries[]` เข้า view model (ตอนนี้ถูกทิ้ง)
- `overlays.ts`: เพิ่ม layer วาด polygon รอยผุ, เลิกวาด `drawBoundingBox`, ปรับสีตาม UX ใหม่
- `FindingsTable.tsx`, `ToothDetailPanel.tsx`: เอา "Detection confidence" ออก เหลือ Probability
- `mocks/resultFactory.ts` + fixture: ใช้ข้อมูล v2

---

## 3. คำถามที่ต้องตกลงกันก่อน freeze

1. เก็บ `teeth[].bbox` ไว้ใน API ไหม (เสนอ: เก็บ เพราะ FE ใช้วาง label และ hit-test) หรือให้ FE คำนวณจาก mask
2. ฟันที่ **ไม่มี** รอยผุยังต้องส่ง mask ฟันมาไหม (คอมเมนต์บอก "ระบุเฉพาะฟันและจุดที่มีรอยผุ")
   เสนอ: ส่งทุกซี่เหมือนเดิม ให้ FE เลือกซ่อนได้ เพราะตาราง/การเลือกฟันยังต้องใช้
3. ชื่อที่แสดงบนเว็บ: "Probability" อย่างเดียว และต้องมีคำอธิบายว่าเป็นคะแนนจากแบบจำลอง ไม่ใช่ความน่าจะเป็นทางคลินิก

---

## 4. บันทึกคอมเมนต์จากการนำเสนอ Proposal (เก็บไว้ทำต่อ ไม่อยู่ใน Sprint นี้)

| สไลด์ | คอมเมนต์ | สิ่งที่ต้องคิดต่อ |
|---|---|---|
| 10 Architecture | ปรับให้เชื่อมต่อในเซิร์ฟเวอร์ภายในคลินิกได้ ใช้งานได้หลายคน | ตอนนี้เป็น single-flight (งานล่าสุดชนะ) → ต้องมี job ต่อผู้ใช้ |
| 10 Architecture | Fetch ถี่ทำให้เปลืองทรัพยากร · การเก็บสถานะควรเป็นแค่ numeric data / log ไม่ควรเก็บเป็น SQL | ทบทวน polling 2 วินาที (เช่น backoff, SSE) และทบทวนการใช้ PostgreSQL เก็บ job state |
| 10 Architecture | ออกแบบ API ใหม่ | เช่น `POST /jobs` → `jobId`, `GET /jobs/:id` แทน `GET /process` แบบไม่ระบุงาน |
| 13 | ใช้ภาษาไปในแนวทางเดียวกันให้เป็นภาพรวม | ชื่อแนวทาง/ศัพท์ในสไลด์ เล่ม และเว็บให้ตรงกัน |
| 14 | นำเสนอสถิติด้วยวิธีที่เหมาะสมมากขึ้น | เลือกตัวชี้วัดหลักและวิธีแสดงผลตาม Evaluation Matrix |
| Demo | Bounding box ซ้ำกับ mask, สีกรอบไม่ผ่าน UX, confidence เกิน, ต้องมี mask รอยผุ | ทำใน Sprint นี้ (§2) |
