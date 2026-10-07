# API Contract v1 — Backend → Frontend

> **สถานะ:** FROZEN (INT-1 / INT-2) · **ฉบับ:** v1 · **วันที่:** 2026-09-30
> **เจ้าของ:** Backend (ไผ่) · **ผู้ใช้หลัก:** Frontend (ปอน) · **แหล่งข้อมูล ML:** เจา
>
> เอกสารนี้คือ spec ของ JSON ที่ Backend ส่งให้หน้าบ้าน **ทุก field** ใช้เอกสารนี้ + fixture
> `backend/src/fixtures/result.sample.json` เป็นตัวตั้งในการทำ mock data และเขียนโค้ด plot ลงรูป

---

## 0. TL;DR สำหรับปอน

- ยิงแค่ 2 endpoint: `POST /process` (อัปโหลด) กับ `GET /process` (poll ทุก ~2 วิ)
- ผลลัพธ์อยู่ใน `data.teeth[]` — ฟัน 1 ซี่ = 1 object มี `bbox`, `mask` (polygon), `axes`, `surfaces`
- **พิกัดทุกตัวเป็น pixel ของรูปต้นฉบับ** (origin มุมซ้ายบน, แกน y ชี้ลง) ไม่มี normalize
- `surfaces` **มีเฉพาะด้านที่ผุ** (0..n รายการ, ชื่อได้แค่ `mesial` / `distal` / `occlusal`)
  ถ้าฟันไม่ผุ → `surfaces: []` และ `has_caries: false` — ไม่มี `sound`, ไม่มี `buccal`/`lingual` แล้ว
- `confidence` ของฟัน = ความมั่นใจว่า **"ตรงนี้คือฟันซี่นี้"** ไม่ใช่ความมั่นใจว่าผุ
- `surfaces[].probability` อาจเป็น `null` (กรณี ML ใช้ fallback) → UI ต้องแสดง "—" แทน %
- Fixture ตัวเต็มอยู่ที่ `backend/src/fixtures/result.sample.json` → copy แบบ byte-identical ไปที่
  `frontend/src/fixtures/result.sample.json`
- สิ่งที่ต้องแก้ใน `inference.ts` / mock ดู [§7](#7-สิ่งที่ปอนต้องแก้-diff-จาก-schema-เดิม)

---

## 1. หลักการ

1. **Backend เป็นเจ้าของ contract นี้** — ML เขียน raw result ลง `/shared/result-{jobId}.json`
   ในรูปแบบของ ML เอง แล้ว Backend (result adapter) validate + แปลงเป็นรูปแบบในเอกสารนี้ก่อนส่งให้ FE
   ดังนั้นถ้าข้างใน ML เปลี่ยน FE ไม่ต้องแก้ ตราบใดที่ contract นี้ไม่เปลี่ยน
2. **Stateless** — ไม่มีการเก็บประวัติ ผลลัพธ์มีแค่ของ job ล่าสุดเท่านั้น
3. **Single-flight** — ประมวลผลทีละ 1 รูป ส่งรูปใหม่ = ยกเลิกงานเก่าอัตโนมัติ (last submission wins)
4. **การเปลี่ยน contract** — ต้องเปิด PR คู่กันเสมอ: `backend/src/routes/process.ts` (+ adapter) ↔
   `frontend/src/domain/inference.ts` และ bump เวอร์ชันเอกสารนี้ ห้ามแก้ฝั่งเดียว

---

## 2. Endpoints

Base URL: `VITE_FRONTEND_API_BASE_URL` (default `http://localhost:8000`)

### 2.1 `POST /process` — ส่งรูป OPG

**Request**

| | |
|---|---|
| Content-Type | `multipart/form-data` (ให้ browser ตั้ง boundary เอง) |
| Field | `image` — ไฟล์เดียว, **ต้องชื่อ `image`** |
| ชนิดไฟล์ | `image/jpeg` หรือ `image/png` |
| ขนาด | ≤ `MAX_IMAGE_MB` (ค่าใน `.env.example` = 25 MB) |
| ขนาดรูป | ≥ `MIN_IMAGE_WIDTH` × `MIN_IMAGE_HEIGHT` px (ค่าใน `.env.example` = 1000 × 500) |

**Responses**

| HTTP | Body | เมื่อไหร่ |
|---|---|---|
| `202` | `{ "status": "processing" }` | รับงานแล้ว → เริ่ม poll `GET /process` |
| `413` | `{ "status": "fail", "fail_message": "Image exceeds 25 MB." }` | ไฟล์ใหญ่เกิน |
| `415` | `{ "status": "fail", "fail_message": "Only JPEG or PNG OPG images are accepted." }` | ไม่ใช่ jpeg/png |
| `422` | `{ "status": "fail", "fail_message": "<เหตุผล>" }` | ไม่มี field `image`, รูปเล็กเกิน, decode ไม่ได้ |
| `503` | `{ "status": "fail", "fail_message": "service temporarily unavailable" }` | ML service ไม่พร้อม / ติดต่อไม่ได้ |

- ทุก error body มี `status: "fail"` + `fail_message` (ข้อความภาษาอังกฤษ พร้อมโชว์ผู้ใช้ได้)
- `202` **ไม่มี** `jobId` ใน body — FE ไม่ต้องรู้ jobId (poll แบบไม่ระบุ id)

### 2.2 `GET /process` — ถามสถานะ/ผลของงานล่าสุด

- ตอบ **HTTP 200 เสมอ** (ยกเว้น server พังจริง ๆ) และมี header `Cache-Control: no-store`
- Body เป็น discriminated union ด้วย field `status`:

| `status` | Body | ความหมาย | FE ควรทำ |
|---|---|---|---|
| `idle` | `{ "status": "idle" }` | ยังไม่เคยส่งรูป (หรือ backend restart) | แสดงหน้า upload |
| `processing` | `{ "status": "processing" }` | กำลังประมวลผล | poll ต่อ |
| `fail` | `{ "status": "fail", "fail_message": "..." }` | งานล่าสุดล้มเหลว | หยุด poll, แสดงข้อความ |
| `done` | `{ "status": "done", "image_base64": "...", "data": { ... } }` | เสร็จ | หยุด poll, render ผล |

- `image_base64` = **data URI เต็ม** พร้อม MIME จริงของไฟล์ที่ส่งมา เช่น
  `data:image/png;base64,iVBOR...` หรือ `data:image/jpeg;base64,/9j/...` → ใช้เป็น `img.src` ได้ทันที
- `done` จะถูกส่งซ้ำได้เรื่อย ๆ จนกว่าจะส่งรูปใหม่ (idempotent)
- งาน ML ใช้เวลาหลายสิบวินาทีต่อรูป (ขึ้นกับ CPU/GPU — ยังไม่ได้วัดจริง) FE ไม่ควรตั้ง timeout สั้น

**Polling:** ทุก `VITE_POLL_INTERVAL_MS` (~2000 ms) ขณะ `processing`; หยุดเมื่อได้ `done` หรือ `fail`
ถ้า network error ให้ retry แบบ backoff (ที่ปอนทำไว้แล้วใน Sprint 7 ใช้ได้เลย)

---

## 3. `data` — อ้างอิงทีละ field

```
data
├── meta
│   ├── job_id        number
│   ├── processed_at  string (ISO-8601 UTC)
│   ├── models        { detector: string, classifier: string }
│   └── timings_ms    Record<string, number>
├── image             { width: number, height: number }
└── teeth[]           Tooth
    ├── id            number
    ├── fdi           number
    ├── confidence    number
    ├── bbox          [x, y, w, h]
    ├── mask          { encoding: "polygon", data: [[x, y], ...] }
    ├── axes          { major: [dx, dy], minor: [dx, dy], rotation_deg: number, clamped: boolean }
    ├── has_caries    boolean
    └── surfaces[]    { name, label, probability, method }
```

### 3.1 `data.meta`

| Field | Type | ตัวอย่าง | ความหมาย / หมายเหตุ |
|---|---|---|---|
| `job_id` | `number` (int) | `1727690400000` | id ของงาน (ใช้ debug/log เท่านั้น ไม่ต้องใช้ใน logic) |
| `processed_at` | `string` | `"2026-09-30T10:00:12.345Z"` | เวลาที่ ML ประมวลผลเสร็จ, ISO-8601 UTC ลงท้าย `Z` เสมอ |
| `models.detector` | `string` | `"unknown"` | เวอร์ชันโมเดล detect ฟัน/ฟันผุ — **ตอนนี้เป็น `"unknown"`** จนกว่า ML จะส่งเวอร์ชันมา (ดู §9) |
| `models.classifier` | `string` | `"unknown"` | เวอร์ชันโมเดลจำแนก surface — เหมือนข้างบน |
| `timings_ms` | `Record<string, number>` | `{}` | เวลาแต่ละ stage (ms) — **ตอนนี้เป็น `{}`** ในอนาคตจะมี key เช่น `detection`, `pca`, `classification` · FE ห้ามสมมติว่ามี key ใด ๆ |

### 3.2 `data.image`

| Field | Type | ความหมาย |
|---|---|---|
| `width` | `number` (int, px) | ความกว้างรูปต้นฉบับ = `naturalWidth` ของ `image_base64` |
| `height` | `number` (int, px) | ความสูงรูปต้นฉบับ = `naturalHeight` ของ `image_base64` |

ใช้คู่นี้เป็น coordinate space ของทุกพิกัดใน `teeth[]`

### 3.3 `data.teeth[]` — ฟันแต่ละซี่

- ความยาว array: `0..32` (อาจเป็น `[]` ได้ถ้าไม่เจอฟันเลย → UI แสดง "ไม่พบฟัน")
- ลำดับใน array: **ไม่รับประกัน** (FE sort เองตามต้องการ เช่นตาม `fdi`)

| Field | Type | ช่วงค่า | ความหมาย / หมายเหตุ |
|---|---|---|---|
| `id` | `number` (int) | `0..n-1` | key ที่ unique ภายในผลลัพธ์นี้ (ใช้เป็น React key / selection / สี) — ไม่ใช่เลขฟัน |
| `fdi` | `number` (int) | `11–18, 21–28, 31–38, 41–48` | เลขฟันระบบ **FDI** (หลักสิบ = quadrant, หลักหน่วย = ตำแหน่ง) ดู §4.4 · **อาจซ้ำกันได้** ถ้าโมเดล detect ซ้ำ (known issue §9) |
| `confidence` | `number` | `0–1` (4 ตำแหน่ง) | ความมั่นใจของโมเดลว่า **"กล่องนี้คือฟันซี่ `fdi`"** (tooth detection) **ไม่ใช่** ความมั่นใจว่าผุ · ถ้าแสดงในตาราง ให้ตั้งชื่อคอลัมน์ว่า "Detection conf." |
| `bbox` | `[number, number, number, number]` | int px | `[x, y, w, h]` — `x,y` = มุมซ้ายบนของกล่อง, `w,h` = กว้าง/สูง (**ไม่ใช่** `x1,y1,x2,y2`) |
| `mask` | `MaskData` | | ขอบเขตรูปร่างฟัน ดู §3.4 |
| `axes` | `ToothAxes` | | แกนหลักของฟันจาก PCA ดู §3.5 |
| `has_caries` | `boolean` | | `true` ⇔ `surfaces.length > 0` (backend คำนวณให้) |
| `surfaces` | `SurfaceFinding[]` | length `0..3` | **เฉพาะด้านที่ตรวจพบฟันผุ** ดู §3.6 |

### 3.4 `mask`

| Field | Type | ความหมาย |
|---|---|---|
| `encoding` | `"polygon"` | v1 ส่งเป็น polygon **เสมอ** (`"rle"` ยังอยู่ใน type เพื่ออนาคต แต่จะไม่ถูกส่ง) |
| `data` | `[number, number][]` | จุด `[x, y]` (float px, พิกัดรูปต้นฉบับ) เรียงตามขอบ — **1 ring, ไม่มีรู** |

- จำนวนจุด ≥ 3 (ถ้าโมเดลไม่ให้ mask จะได้ 4 จุด = มุม bbox) — จำนวนจุดไม่คงที่ ขึ้นกับรูปร่างฟัน
- **ring ไม่ปิด** — จุดสุดท้ายไม่ซ้ำจุดแรก → ใช้ `ctx.closePath()`
- polygon อาจล้นออกนอก `bbox` ได้เล็กน้อย (bbox กับ mask มาจากคนละขั้นตอน) อย่าใช้ bbox clip mask

### 3.5 `axes`

| Field | Type | ความหมาย |
|---|---|---|
| `major` | `[dx, dy]` | unit vector แกนยาวของฟัน (ความแปรปรวนมากสุด) |
| `minor` | `[dx, dy]` | unit vector แกนสั้น (ตั้งฉากกับ `major`) |
| `rotation_deg` | `number` | องศาที่ต้องหมุนฟันให้ตั้งตรง (2 ตำแหน่ง), ช่วง `-45..45` |
| `clamped` | `boolean` | `true` = มุมเดิมเกิน 45° จึงถูกตั้งเป็น `0` (ความเชื่อถือต่ำ) |

- **เครื่องหมายของ vector สุ่มได้** (`[0.02, 0.99]` กับ `[-0.02, -0.99]` คือแกนเดียวกัน) → วาดเป็นเส้นผ่านจุดศูนย์กลาง
  ทั้งสองทิศ ไม่ใช่ลูกศร
- จุดศูนย์กลางไม่ได้ส่งมา → ใช้ centroid ของ `mask.data` (หรือกลาง `bbox`)
- **PCA ล้มเหลว** → `major = minor = [0, 0]`, `rotation_deg = 0`, `clamped = false` → FE **ไม่ต้องวาดแกน**
  (เช็คด้วย `major[0] === 0 && major[1] === 0`)

### 3.6 `surfaces[]` — ด้านที่ผุ

| Field | Type | ค่าที่เป็นไปได้ | ความหมาย |
|---|---|---|---|
| `name` | `string` enum | `"mesial"` \| `"distal"` \| `"occlusal"` | ด้านของฟันที่ผุ · **ไม่ซ้ำกันภายในฟันซี่เดียว** (ใช้เป็น React key ได้) |
| `label` | `"caries"` | `"caries"` เท่านั้น | คงไว้เพื่อ forward-compat — v1 มีแต่ด้านที่ผุ |
| `probability` | `number \| null` | `0–1` หรือ `null` | ความมั่นใจของ RF ว่า **"ผุที่ด้านนี้"** (เทียบกับอีก 2 ด้าน) · `null` เมื่อ `method = "XThirds_Fallback"` |
| `method` | `string` enum | `"RF"` \| `"XThirds_Fallback"` | `RF` = Random Forest จำแนก · `XThirds_Fallback` = แบ่งฟันเป็น 3 ส่วนตามแนว x (ใช้เมื่อ RF ใช้ไม่ได้) ความเชื่อถือต่ำกว่า |

- ความยาว: ปัจจุบัน ML ให้ **0 หรือ 1** รายการต่อซี่ แต่ FE ต้องรองรับ `0..3`
- ถ้าจะแสดง "ผุ x ด้าน" ให้ใช้ตัวหารคงที่ **3** (`"1 / 3 surfaces"`) ไม่ใช่ `surfaces.length`
- **ไม่มีตำแหน่งพิกัดของรอยผุ** ใน v1 — FE ไฮไลต์ได้ระดับ "ฟันซี่นี้ผุด้าน X" (เช่น tint ทั้ง mask สีแดง
  + label ด้าน) ถ้าต้องการวาดโซน mesial/distal/occlusal บนฟัน ให้คำนวณจาก `axes` + `mask` ฝั่ง FE
  (แกน minor แบ่งซ้าย/ขวา = mesial/distal, ปลายแกน major ฝั่งด้านบดเคี้ยว = occlusal)

---

## 4. Convention สำหรับการวาดบน Canvas

### 4.1 Coordinate space
```
(0,0) ──────────── x → ──────────── (image.width, 0)
  │
  y ↓           ┌──────┐  bbox = [x, y, w, h]
  │     (x, y)  │  🦷  │ h
  │             └──────┘
  │                w
(0, image.height)
```
- วาดด้วย transform เดียวกับรูป: `ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY)` แล้ว
  ใช้พิกัดจาก JSON ตรง ๆ ได้เลย (ไม่ต้อง scale ทีละจุด)
- hit-test (คลิกเลือกฟัน): แปลงจุดเมาส์กลับเป็น image space แล้วใช้ point-in-polygon กับ `mask.data`
  (`isPointInPolygon` ใน `lib/rle.ts` ใช้ได้เลย) — ถ้าทับกันหลายซี่ เลือกซี่ที่ bbox เล็กสุด

### 4.2 ลำดับการวาดที่แนะนำ
1. รูป (พร้อม brightness/contrast filter) → reset filter
2. mask ของทุกซี่ (fill โปร่ง ~25%) — ซี่ที่ `has_caries` ใช้สีแดง ที่เหลือใช้สี identity
3. bbox (stroke บาง)
4. axes (เส้น major/minor ผ่าน centroid)
5. label `fdi` (+ ชื่อด้านที่ผุ) ที่มุม bbox
6. selection halo ของซี่ที่เลือก (บนสุด)

### 4.3 ช่วงความยาวเส้นแกน
- ความยาวแนะนำ: `major` ≈ `0.5 × max(w, h)` ต่อข้าง, `minor` ≈ `0.5 × min(w, h)` ต่อข้าง

### 4.4 FDI chart (มุมมองในภาพ OPG)
```
          ขวาผู้ป่วย (ซ้ายภาพ)          ซ้ายผู้ป่วย (ขวาภาพ)
บน   18 17 16 15 14 13 12 11 | 21 22 23 24 25 26 27 28
ล่าง  48 47 46 45 44 43 42 41 | 31 32 33 34 35 36 37 38
```
- Quadrant: 1 = บนขวา, 2 = บนซ้าย, 3 = ล่างซ้าย, 4 = ล่างขวา (ของผู้ป่วย)
- ฟันบน: `fdi` 11–28 · ฟันล่าง: `fdi` 31–48 (`Math.floor(fdi / 10) <= 2` = ฟันบน)
- 1–3 = ฟันหน้า (incisor/canine), 4–5 = premolar, 6–8 = molar

---

## 5. ตัวอย่าง `done` response

ไฟล์เต็ม (4 ซี่, valid JSON): **`backend/src/fixtures/result.sample.json`** · ด้านล่างตัด polygon ให้สั้นลง

ครอบคลุม 4 กรณี: ฟันไม่ผุปกติ (16) · ฟันไม่ผุ + PCA ล้มเหลว (11) · ผุแบบ RF (36) · ผุแบบ fallback (46)

```jsonc
{
  "status": "done",
  "image_base64": "data:image/png;base64,iVBORw0KGgo...",   // fixture ใช้รูป 1x1 placeholder
  "data": {
    "meta": {
      "job_id": 1727690400000,
      "processed_at": "2026-09-30T10:00:12.345Z",
      "models": { "detector": "unknown", "classifier": "unknown" },
      "timings_ms": {}
    },
    "image": { "width": 3036, "height": 1536 },
    "teeth": [
      {
        "id": 0, "fdi": 16, "confidence": 0.9412,
        "bbox": [760, 480, 190, 290],
        "mask": { "encoding": "polygon", "data": [[947.2, 625], [940.5, 686.7], "... 12 จุด"] },
        "axes": { "major": [0.0213, 0.9998], "minor": [0.9998, -0.0213], "rotation_deg": -1.22, "clamped": false },
        "has_caries": false,
        "surfaces": []
      },
      {
        "id": 1, "fdi": 11, "confidence": 0.9687,
        "bbox": [1452, 430, 118, 330],
        "mask": { "encoding": "polygon", "data": ["..."] },
        "axes": { "major": [0, 0], "minor": [0, 0], "rotation_deg": 0, "clamped": false },   // PCA fail → ไม่วาดแกน
        "has_caries": false,
        "surfaces": []
      },
      {
        "id": 2, "fdi": 36, "confidence": 0.9135,
        "bbox": [2080, 820, 205, 280],
        "mask": { "encoding": "polygon", "data": ["..."] },
        "axes": { "major": [-0.0871, 0.9962], "minor": [0.9962, 0.0871], "rotation_deg": 5, "clamped": false },
        "has_caries": true,
        "surfaces": [
          { "name": "occlusal", "label": "caries", "probability": 0.815, "method": "RF" }
        ]
      },
      {
        "id": 3, "fdi": 46, "confidence": 0.8876,
        "bbox": [790, 830, 200, 275],
        "mask": { "encoding": "polygon", "data": ["..."] },
        "axes": { "major": [0.1219, 0.9925], "minor": [0.9925, -0.1219], "rotation_deg": -7, "clamped": false },
        "has_caries": true,
        "surfaces": [
          { "name": "mesial", "label": "caries", "probability": null, "method": "XThirds_Fallback" }
        ]
      }
    ]
  }
}
```

> **หมายเหตุ fixture:** `image_base64` เป็นรูป 1×1 (ทดสอบ schema เท่านั้น) ถ้าจะ demo ภาพจริงให้ mock
> factory ใช้รูปที่อัปโหลดจริง แล้ววาง polygon ภายใน `image.width × image.height` ของรูปนั้น
> (รูป OPG ตัวอย่าง: `reserch/images/raw_panoramic_xray.png` ขนาด 3036×1536 ตรงกับ fixture)

---

## 6. Zod schema v1 (พร้อม copy ไปใส่ `frontend/src/domain/inference.ts`)

```ts
import { z } from 'zod';

export const SURFACE_NAMES = ['mesial', 'distal', 'occlusal'] as const;
export const SURFACE_COUNT = SURFACE_NAMES.length; // 3 — ใช้เป็นตัวหาร "x / 3 surfaces"

const point = z.tuple([z.number(), z.number()]);

const surfaceFindingSchema = z.object({
  name: z.enum(SURFACE_NAMES),
  label: z.literal('caries'),
  probability: z.number().min(0).max(1).nullable(),
  method: z.enum(['RF', 'XThirds_Fallback']),
});

const toothAxesSchema = z.object({
  major: point,
  minor: point,
  rotation_deg: z.number(),
  clamped: z.boolean(),
});

const maskDataSchema = z.discriminatedUnion('encoding', [
  z.object({ encoding: z.literal('polygon'), data: z.array(point).min(3) }),
  z.object({ encoding: z.literal('rle'), data: z.array(z.number()) }), // ไม่ถูกส่งใน v1
]);

const toothSchema = z.object({
  id: z.number().int(),
  fdi: z.number().int().min(11).max(48),
  confidence: z.number().min(0).max(1),
  bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]), // x, y, w, h
  mask: maskDataSchema,
  axes: toothAxesSchema,
  has_caries: z.boolean(),
  surfaces: z.array(surfaceFindingSchema).max(3),
});

const inferenceMetaSchema = z.object({
  job_id: z.number().int(),
  processed_at: z.string(),
  models: z.object({ detector: z.string(), classifier: z.string() }),
  timings_ms: z.record(z.string(), z.number()),
});

const inferenceDataSchema = z.object({
  meta: inferenceMetaSchema,
  image: z.object({ width: z.number().int().positive(), height: z.number().int().positive() }),
  teeth: z.array(toothSchema),
});

export const processResponseSchema = z.discriminatedUnion('status', [
  z.object({ status: z.literal('idle') }),
  z.object({ status: z.literal('processing') }),
  z.object({ status: z.literal('fail'), fail_message: z.string() }),
  z.object({ status: z.literal('done'), image_base64: z.string(), data: inferenceDataSchema }),
]);

export type ProcessResponse = z.infer<typeof processResponseSchema>;
export type InferenceData = z.infer<typeof inferenceDataSchema>;
export type Tooth = z.infer<typeof toothSchema>;
export type SurfaceFinding = z.infer<typeof surfaceFindingSchema>;
export type ToothAxes = z.infer<typeof toothAxesSchema>;
export type MaskData = z.infer<typeof maskDataSchema>;

export function parseProcessResponse(json: unknown): ProcessResponse {
  return processResponseSchema.parse(json);
}
```

> Backend จะใช้ schema ตัวเดียวกันนี้ validate ผลก่อนส่ง — ถ้า ML ส่งของผิดรูปมา backend จะตอบ
> `{status:"fail", fail_message:"Invalid inference result."}` แทนการส่งข้อมูลพัง ๆ ให้ FE

---

## 7. สิ่งที่ปอนต้องแก้ (diff จาก schema เดิม)

| จุด | เดิม (`inference.ts` ปัจจุบัน) | v1 | ไฟล์ที่กระทบ |
|---|---|---|---|
| `surfaces[].name` | 5 ค่า (+ `buccal`, `lingual`) | 3 ค่า `mesial`/`distal`/`occlusal` | `inference.ts`, `resultFactory.ts` |
| `surfaces[].label` | `caries` \| `sound` | `"caries"` เท่านั้น | `ToothDetailPanel.tsx` (ไฮไลต์ทุกแถว) |
| `surfaces` ความหมาย | ครบทุกด้าน | **เฉพาะด้านที่ผุ** (0..3) | `analysisTypes.ts`, `ToothDetailPanel.tsx` (แสดงแถว 3 ด้านเอง หรือแสดงเฉพาะที่ผุ) |
| `surfaces[].probability` | `number` | `number \| null` | `ToothDetailPanel.tsx:68` — `null` → แสดง `"—"` |
| `surfaces[].method` | – | **ใหม่** `"RF"` \| `"XThirds_Fallback"` | แสดง badge "low confidence" ถ้าเป็น fallback (optional) |
| `teeth[].has_caries` | – (คำนวณใน FE) | **ใหม่** มาจาก backend | `analysisTypes.ts` ใช้ได้ตรง ๆ |
| `axes.clamped` | – | **ใหม่** `boolean` | overlays: ถ้า `major` เป็น `[0,0]` ไม่ต้องวาด |
| `meta.job_id` | – | **ใหม่** | – (ไม่ต้องใช้) |
| `meta.models`, `timings_ms` | มีค่าจริง | ตอนนี้ `"unknown"` / `{}` | ถ้าแสดง ให้รองรับค่าว่าง |
| `cariesSummary` | `` `${n} / ${surfaces.length}` `` | `` `${n} / 3 surfaces` `` (ใช้ `SURFACE_COUNT`) | `analysisTypes.ts:73` |
| `totalCariesSurfaces` | นับ label caries | = ผลรวม `surfaces.length` | `analysisTypes.ts` |
| `teeth[].id` | เริ่ม 1 | เริ่ม **0** | ไม่กระทบ (แค่ key) |
| `fdi` | unique | **อาจซ้ำ** (known issue) | ห้ามใช้ `fdi` เป็น React key → ใช้ `id` |
| fixture | ของเดิม | copy จาก `backend/src/fixtures/result.sample.json` | `frontend/src/fixtures/` + `inference.test.ts` |
| mock factory | 5 ด้าน + sound | สุ่ม 0–1 ด้านที่ผุ, บางซี่เป็น fallback (`probability: null`), บางซี่ axes `[0,0]` | `mocks/resultFactory.ts` |

---

## 8. ML raw → API v1 mapping (สำหรับ backend adapter)

Raw ที่ ML เขียน: ดูตัวอย่าง `backend/src/fixtures/ml-result.raw.sample.json`
(โค้ดต้นทาง: `ml-service/app/pipeline/orchestrator.py`, `postprocess.py`, `surface_classification.py`)

| API v1 | จาก ML raw | กฎการแปลง |
|---|---|---|
| `meta.job_id` | `meta.job_id` | ตรง ๆ |
| `meta.processed_at` | `meta.completed_at` | `new Date(x).toISOString()` (normalize เป็น UTC `Z`, ms 3 หลัก) |
| `meta.models` | – | `{detector:"unknown", classifier:"unknown"}` จนกว่า ML จะส่ง `meta.models` มา แล้วใช้ของ ML |
| `meta.timings_ms` | – | `{}` จนกว่า ML จะส่ง `meta.timings_ms` มา |
| `image.width/height` | `meta.image_size.width/height` | ตรง ๆ |
| – | `meta.tooth_count`, `meta.caries_count` | ทิ้ง (FE คำนวณเอง) |
| `teeth[].id` | – | **index ใน array** (ไม่เชื่อ `id` จาก ML) |
| `teeth[].fdi` | `teeth[].fdi` | ตรง ๆ |
| `teeth[].confidence` | `teeth[].confidence` | ตรง ๆ (pano YOLO tooth conf) |
| `teeth[].bbox` | `teeth[].bbox` | ตรง ๆ (`[x,y,w,h]` int) |
| `teeth[].mask` | `teeth[].mask` | ตรง ๆ; ถ้า < 3 จุด → ใช้ 4 มุม bbox |
| `teeth[].axes` | `teeth[].axes` | ถ้าไม่มี `clamped` → `false` |
| `teeth[].has_caries` | – | `surfaces.length > 0` |
| `surfaces[].name` | `surfaces[].name` | ตรง ๆ (lowercase แล้ว); dedupe ถ้าชื่อซ้ำ (เก็บตัวที่ probability สูงกว่า) |
| `surfaces[].label` | `surfaces[].label` | `"caries"` |
| `surfaces[].probability` | `surfaces[].probability` | `method === "RF"` → ค่าเดิม, ไม่งั้น → `null` |
| `surfaces[].method` | `surfaces[].method` | ตรง ๆ |
| – | `surfaces[].fallback_reason` | ทิ้ง (log ฝั่ง backend) |

ถ้า raw ไม่ผ่าน schema (key หาย / type ผิด) → `GET /process` ตอบ
`{status:"fail", fail_message:"Invalid inference result."}` และ log รายละเอียดฝั่ง backend

---

## 9. Known issues / สิ่งที่ต้องให้เจา (ML) แก้

| # | ปัญหา | ที่อยู่ | ผลกระทบ | ความเร่งด่วน |
|---|---|---|---|---|
| ML-1 | `build_tooth_result()` ต้องการ `id` แต่ caller ไม่ส่ง → `TypeError` | `orchestrator.py` / `postprocess.py` | **แก้แล้ว** — orchestrator ส่ง `id` จาก index ของ detection | ✅ |
| ML-2 | ผล Stage 2/3 เก็บใน dict keyed ด้วย `fdi` + ไม่มี IoU dedupe ฟันซ้ำ | `pca_alignment.py:181`, `surface_classification.py:285`, `detection.py:76-106` | ฟันซ้ำ FDI ได้ axes/surfaces ทับกัน, "All detected teeth" มีซี่ซ้ำ | 🟠 สูง (key ด้วย index + dedupe IoU>0.5 แบบ `reserch/week4/inference.py:474`) |
| ML-3 | `CARIES_CONF=0.005` รับแทบทุก box | `config.py`, `docker-compose.yml` | **แก้เบื้องต้น** — ค่า default/runtime เปลี่ยนเป็น `0.02`; ต้องประเมิน detector เพิ่มเพราะ threshold ไม่ได้แก้ model recall/precision ทั้งหมด | 🟡 ตรวจต่อ |
| ML-4 | RF ได้ input เป็น vertex ของ polygon + grid จุดของ caries box แทน pixel coordinates แบบตอนเทรน | `surface_classification.py:182,246`, `detection.py:116-122` | feature `coverage` ผิดสเกล → surface/probability ไม่น่าเชื่อ | 🟠 สูง |
| ML-5 | fallback ใส่ `probability: 0.0` แต่ label ยังเป็น caries | `surface_classification.py:197,223` | backend แปลงเป็น `null` ให้แล้ว (ไม่ต้องแก้ด่วน) | 🟢 ต่ำ |
| ML-6 | ไม่ส่ง model versions / timings | `orchestrator.py:106` (มีข้อมูลใน `models/versions.py`) | `meta.models`="unknown", `timings_ms`={} | 🟢 ต่ำ — ขอให้ส่ง `meta.models:{detector, classifier}` และ `meta.timings_ms:{detection, pca, classification}` |
| ML-7 | `/health` ตอบ `ready:true` เสมอ, `verify_artifacts()` ไม่ถูกเรียก | `main.py:52-58` | backend รู้ไม่ได้ว่า weights พร้อมไหม | 🟢 ต่ำ |
| ML-8 | axes default ตอน PCA fail ไม่มี `clamped` | `postprocess.py` `safe_axes` | backend เติม `false` ให้แล้ว | 🟢 ต่ำ |
| ML-9 | `meta.caries_count` นับจำนวน "ฟัน" ไม่ใช่ surface | `orchestrator.py:112` | backend ทิ้ง field นี้ | 🟢 ต่ำ |

> เรื่อง "Confidence" ใน UI: `teeth[].confidence` คือความมั่นใจการ detect ฟัน ส่วนความมั่นใจของ
> caries YOLO ยังไม่ถูกส่งออกมา (ตกลงกันแล้วว่า v1 ใช้เท่าที่มี) ถ้าอนาคตจะเพิ่ม `caries_confidence`
> ให้ทำเป็น v1.1 (เพิ่ม field แบบ optional ไม่ทำให้ของเดิมพัง)

---

## 10. Changelog

| เวอร์ชัน | วันที่ | เปลี่ยนแปลง |
|---|---|---|
| v1 | 2026-09-30 | ฉบับแรกที่ freeze — surfaces เหลือเฉพาะด้านที่ผุ (3 ด้าน), เพิ่ม `has_caries`, `method`, `axes.clamped`, `meta.job_id`; `probability` nullable; backend adapter แปลงผล ML |
