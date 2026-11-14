# Frontend — ภาคเรียน 2 Sprint 1 (ปอน) · 2026-11-14

> วันที่/เวลาดำเนินการ: **2026-11-14** · รัน typecheck/lint/test/build รอบสุดท้ายเมื่อ 21:04 (UTC+7)
> ผู้รับผิดชอบ: ปอน (Frontend) · Branch: `feat/sprint_1_sem2_be`
> อ้างอิง: [API contract v2 draft](../api-contract-v2-draft.md) §2.1 / §2.4 · [Progress Backend ของไผ่](../be-sem2-sprint1-progress-2026-11-14.md)
> สถานะ: commit และ push ขึ้น `feat/sprint_1_sem2_be` แล้ว (รอบที่ 1 + รอบที่ 2 อยู่ใน commit เดียวกัน; ดูหัวข้อ 5)

ขอบเขต: แก้เฉพาะ `dental-caries-detection/frontend/` เท่านั้น — `backend/`, `ml-service/`, `docs-md/` (นอกจากไฟล์นี้) ไม่ถูกแก้
(ตรวจด้วย `git diff --stat -- backend docs-md ml-service` ได้ 0 ไฟล์)

---

## 1. สรุปสิ่งที่ทำในรอบนี้

1. อ่าน `docs-md/` (v2 draft, v1.2, progress BE), `backend/` (`resultAdapter.ts`, `routes/process.ts`, fixtures, tests) และโค้ด FE ทั้งหมดที่เกี่ยวข้อง
   ก่อนลงมือ — สเปคใน docs ตรงกับโค้ด adapter ของไผ่ ไม่พบจุดขัดแย้งที่ต้องหยุดถาม
2. **Wire schema → API v2 draft:** ตัด `teeth[].confidence`, เปลี่ยน `caries[]` เป็น `{ id, surface|null, probability, mask, centroid }` และบังคับให้มีทุกซี่ (`[]` ถ้าไม่มี)
3. **View-model:** ส่ง `caries[]` เข้า `ToothViewModel` (เดิมถูกทิ้ง), ตัด `confidence` ออก
4. **Canvas:** เพิ่ม layer "Caries Mask" (ปุ่มเปิด/ปิด, default เปิด) วาด polygon รอยผุทับบนซี่ฟัน; **เลิกวาด bounding box** (ลบ layer "Boxes" และ `drawBoundingBox`), selection halo เปลี่ยนจากกรอบสี่เหลี่ยมรอบ bbox เป็นเส้นขอบตาม mask ของฟัน
5. **UI:** ตาราง "All detected teeth" ตัดคอลัมน์ Confidence; แผงรายละเอียดฟันตัด "Detection confidence" และเพิ่มตาราง "Caries lesions" (Lesion / Surface / Probability) โดย `surface = null` แสดง "Unspecified"
6. **Mocks:** `resultFactory.ts` สร้างข้อมูลตาม v2 (มี lesion mask 2 จุดบนซี่ที่ผุ จุดหนึ่ง `surface: null`)
7. **Fixture:** คัดลอก `backend/src/fixtures/result.sample.json` → `frontend/src/fixtures/result.sample.json` แบบ byte-identical (ตรวจด้วย `cmp`) ตามกฎ contract v1 §0 — ปิดงานค้างข้อ 4 ในรายงานของไผ่
8. **Tests:** เขียนใหม่/เพิ่ม ครอบคลุม schema v2, view-model, overlay primitive, การสลับ layer จริงผ่าน `CanvasViewer`
9. ปรับข้อความที่ผู้ใช้เห็น (How to use, subtitle, README) ให้ตรงกับ UI ใหม่
10. **รอบที่ 2 (defensive guard ฝั่ง FE ล้วน, ไม่แตะ backend):** ผ่อน schema ให้ `caries[].mask.data` ว่าง/1/2 จุดก็ parse ผ่าน, ข้ามการวาด lesion ที่มี < 3 จุด, ลบ fixture v1 เก่า — ดูหัวข้อ 5

---

## 2. ไฟล์ที่แก้ไข / สร้างใหม่ (ทั้งหมดใน `frontend/`)

### แก้ไข

| ไฟล์ | จุดที่แก้ |
|---|---|
| `src/domain/inference.ts` | **`CariesLesion`**: เอา `bbox`, `confidence` ออก → `{ id, surface: SurfaceName \| null, probability, mask, centroid }`; เพิ่ม type `SurfaceName`. **`Tooth`**: ลบ `confidence`, `caries` จาก optional → required. **`cariesLesionSchema`** (Zod): `id` int ≥ 0, `surface` enum 3 ค่า `.nullable()`, `probability` 0–1, `mask` polygon (รอบที่ 2: ไม่บังคับขั้นต่ำจุดแล้ว), `centroid`. **`toothSchema`**: ลบ `confidence`, `caries: z.array(cariesLesionSchema)` (ไม่ `.optional()`) |
| `src/features/analysis/analysisTypes.ts` | **`ToothViewModel`**: ลบ `confidence`, เพิ่ม `caries: CariesLesion[]`. **`toToothViewModel()`**: ลบ `confidence: tooth.confidence`, เพิ่ม `caries: tooth.caries`. อัปเดตคอมเมนต์ palette. `toViewModel()` / summary ไม่เปลี่ยนตรรกะ |
| `src/components/CanvasViewer/overlays.ts` | **ลบ** `drawBoundingBox` และค่า `SELECTION_HALO_PADDING`. **เพิ่ม** helper `tracePolygon()` (ใช้ร่วมกัน), **`drawCariesMasks(ctx, tooth)`**, ค่า export `CARIES_MASK_FILL` / `CARIES_MASK_STROKE`. **`drawMask`** ใช้ `tracePolygon` (พฤติกรรมเดิม). **`drawSelectionHalo`** เปลี่ยน signature เป็น `(ctx, tooth, imgWidth, imgHeight)` และ stroke ตาม polygon ของ tooth mask แทน `strokeRect` |
| `src/components/CanvasViewer/layerState.ts` | `LayerVisibility`: ลบ `boxes`, เพิ่ม `caries` (default `true`); `setAll()` ปรับให้ตรง |
| `src/components/CanvasViewer/CanvasViewer.tsx` | `LAYER_LABELS`: ลบ `boxes: 'Boxes'`, เพิ่ม `caries: 'Caries Mask'` (checkbox render จาก `Object.keys(layers)` จึงขึ้นอัตโนมัติ ลำดับ Masks → Caries Mask → Axes → Labels) |
| `src/components/CanvasViewer/useCanvasRenderer.ts` | draw loop: ลบ pass `boxes`, เพิ่ม pass `if (layers.caries) drawCariesMasks(...)` ต่อจาก pass tooth masks; ส่ง `imageSize` ให้ `drawSelectionHalo`. **hit-test ไม่เปลี่ยน** (ยังใช้ mask แล้ว fallback เป็น `tooth.bbox` ตามที่ v2 draft ให้คงไว้) |
| `src/components/FindingsTable.tsx` | ลบคอลัมน์ "Confidence" (`<th>` + `<td>`), empty-state `colSpan` 3 → 2 |
| `src/components/ToothDetailPanel.tsx` | ข้อความหัวแผงจาก `Detection confidence: N% · …` → แสดงแค่ `cariesSummary`; เพิ่ม helper `formatProbability()` (ใช้ซ้ำกับตาราง surface เดิม); เพิ่มตาราง **"Caries lesions"** เมื่อ `caries.length > 0` — แถวละ lesion: `#id+1`, surface (`null` → "Unspecified" สีเทา), Probability % |
| `src/features/analysis/AnalysisView.tsx` | subtitle "AI detected findings and confidence scores." → "… probability scores." |
| `src/data/siteContent.ts` | ข้อความ How to use: ตัดคำว่า bounding box / Boxes toggle, อธิบายสีเหลือง = รอยผุ, toggle ใหม่ Masks/Caries Mask/Axes/Labels, ตัด "detection confidence" |
| `src/mocks/resultFactory.ts` | สร้างข้อมูลแบบ v2: ลบ `confidence`; `buildSurfaces()` → `buildCaries()` คืน `{ surfaces, caries }`; ซี่ที่ผุมี lesion 2 จุด (id 0 = occlusal, probability เท่ากับ surface; id 1 = `surface: null`); เพิ่ม `blobAt()`, `centroidOf()`; `method` 'RF' → 'Geometry_Heuristic' |
| `src/mocks/handlers.ts` | แก้คอมเมนต์ API v1 → v2 draft (ไม่เปลี่ยนโค้ด) |
| `README.md` | ย่อหน้า View-model layer: ระบุ `caries[]` และตัดคำว่า bounding-box stroke |

### สร้างใหม่

| ไฟล์ | หน้าที่ |
|---|---|
| `src/fixtures/result.sample.json` | สำเนา byte-identical ของ `backend/src/fixtures/result.sample.json` (case 10: 29 ซี่, lesion 2 จุดที่ FDI 45 และ 26 ด้าน distal) |
| `src/test-utils/v2Fixture.ts` | `v2DoneResponse` และ `loadV2Data()` ให้เทสต์ใช้ fixture v2 ร่วมกัน |
| `src/components/CanvasViewer/__tests__/overlays.test.ts` | เทสต์ primitive วาด |
| `src/components/CanvasViewer/__tests__/CanvasViewer.test.tsx` | เทสต์การสลับ layer ผ่าน component จริง |

### เขียนทดสอบใหม่ / ปรับ

`src/domain/__tests__/inference.test.ts`, `src/features/analysis/__tests__/analysisTypes.test.ts`,
`src/components/__tests__/FindingsTable.test.tsx`, `src/components/__tests__/ToothDetailPanel.test.tsx`,
`src/components/CanvasViewer/__tests__/layerState.test.ts` (รายละเอียดข้อ 4)

> `src/fixtures/ml-result.raw.sample.json` (fixture v1 เดิม) ถูกลบในรอบที่ 2 (ดูหัวข้อ 5)

---

## 3. รายละเอียดทางเทคนิคและผลกระทบ

### 3.1 Schema (wire) — breaking change

| Field | v1.x (เดิม) | v2 draft (ตอนนี้) |
|---|---|---|
| `teeth[].confidence` | required | **ไม่มี** (ถ้า backend ส่งมา Zod จะ strip ทิ้ง) |
| `teeth[].bbox` | required | คงไว้ (ใช้ anchor label + hit-test fallback, ไม่วาดแล้ว) |
| `teeth[].caries` | optional | **required** — ไม่มี key นี้ = parse fail (backend ส่ง `[]` เสมอ) |
| `caries[].id` | – | ใหม่ int ≥ 0 |
| `caries[].surface` | – | ใหม่ `mesial\|distal\|occlusal\|null` (ต้องมี key; backend ส่ง `null` ถ้า ML-10 ยังไม่มา) |
| `caries[].confidence` → `probability` | `confidence` | เปลี่ยนชื่อ; payload เก่าที่ใช้ `confidence` จะ **ถูกปฏิเสธ** |
| `caries[].bbox` | required | **ไม่มี** (strip) |
| `caries[].mask` / `centroid` | มี | `centroid` เหมือนเดิม; `mask.data` รอบที่ 2 รับ polygon กี่จุดก็ได้ (รวมว่าง) แต่แต่ละจุดต้องเป็น `[x, y]` |

ผลกระทบ: ถ้า FE นี้ไปต่อกับ backend รุ่นเก่า (v1.x ที่ไม่มี `probability`/ไม่มี `caries`) หน้าเว็บจะ parse fail และแสดง error — ต้อง deploy คู่กับ backend ที่ไผ่ทำ (ตามกฎ "เปิด PR คู่" ใน contract v1 §1.4)
`surfaces[]`, `meta`, `image`, `axes`, `has_caries` ไม่เปลี่ยน

### 3.2 View-model

`ToothViewModel` ตัด `confidence` และเพิ่ม `caries: CariesLesion[]` (อ้างอิง array เดิม ไม่ copy) — `cariesCount` / `cariesSummary` / `hasCaries` / `summary` ยังนับจาก **`surfaces[]`** เหมือนเดิม ไม่ใช่จำนวน lesion จึงไม่กระทบตัวเลขสรุปเดิม

### 3.3 Canvas

- พิกัด lesion อยู่ใน pano pixel space เดียวกับ tooth mask → วาดภายใต้ transform pan/zoom เดิมของ renderer (`translate` + `scale`) ได้ตรง ๆ ไม่มีการคำนวณพิกัดเพิ่ม และไม่แตะ hit-test/zoom/pan
- ลำดับวาด: รูป → tooth masks → **caries masks** → axes → labels → selection halo (บนสุด) ทำให้ lesion อยู่เหนือสีฟันแต่ใต้ label/halo
- layer `caries` แยกอิสระจาก layer `masks` — ปิด tooth masks แล้ว lesion ยังแสดง (มีเทสต์)
- **สี (ตัดสินใจเอง ตามที่ให้อิสระไว้):** tooth ที่ผุยังเป็นแดงโปร่ง `rgba(220,38,38,.35)`, ฟันปกติฟ้าโปร่ง; lesion ใช้ **เหลืองอำพัน** `rgba(250,204,21,.6)` + ขอบ `#fef08a` กว้าง 3 px (image-space) — ตั้งใจไม่ใช้แดงเพื่อไม่กลืนกับสีฟันที่ผุ และขอบทึบช่วยให้ lesion เล็กยังเห็นเมื่อ fit-to-view
- Selection halo: เดิมสี่เหลี่ยมรอบ bbox (เท่ากับยังเห็นกรอบ bbox บนจอเมื่อคลิกฟัน) → ตอนนี้ stroke ตาม polygon ของฟัน สีส้มเดิม + glow เดิม
- คำที่ยังอ้างถึง bbox ใน FE: `drawLabel` (ตำแหน่ง label), `drawAxes` (จุดกึ่งกลาง), hit-test fallback — ไม่ได้วาดกรอบ

### 3.4 Edge cases ที่จัดการ

| กรณี | พฤติกรรม |
|---|---|
| `surface: null` | ตาราง lesion แสดง "Unspecified"; canvas ไม่ขึ้นกับ surface; ไม่ crash |
| `caries: []` | ไม่วาดอะไร; ไม่แสดงตาราง lesion |
| surface probability `null` | แสดง "—" เหมือนเดิม |
| lesion ID | ใช้ `id` เป็น React key (unique ต่อซี่ตาม contract) |

### 3.5 ข้อสังเกต / ความเสี่ยงที่ควรแจ้งทีม

1. ~~`caries[].mask.data` ต้อง ≥ 3 จุดฝั่ง FE~~ — **แก้แล้วในรอบที่ 2** (หัวข้อ 5): FE ไม่ fail ทั้งงานอีก; adapter ของไผ่ยังไม่ตรวจ polygon ของ lesion แต่ไม่ต้องแก้ฝั่ง backend แล้ว
2. `image_base64` ใน fixture เป็น 1×1 placeholder → **ยังไม่ได้ตรวจการวางตำแหน่ง mask บนภาพ X-ray จริง** และไม่ได้เปิดดูผลใน browser (ตรวจผ่านเทสต์ + build เท่านั้น); ควรลอง end-to-end กับ backend + ML จริงเพื่อดูความสวยงามของสี/ความหนาเส้นอีกครั้ง
3. ข้อที่ยังเปิดอยู่ใน v2 draft §3 (ยังไม่ปรับใน FE): คำอธิบายใต้ Probability ว่าเป็นคะแนนโมเดลที่ยังไม่ calibrated — draft ยังไม่ตกลง จึงยังไม่ได้เพิ่มข้อความ
4. ชื่อปุ่ม "Caries Mask" และข้อความ "Unspecified" เป็นการเลือกของ FE (draft ใช้คำว่า "ไม่ระบุ" แต่ UI ทั้งหมดเป็นภาษาอังกฤษ) — ปรับได้ที่ `CanvasViewer.tsx` (`LAYER_LABELS`) และ `ToothDetailPanel.tsx`

---

## 4. Tests

| ไฟล์ | สิ่งที่ครอบคลุม |
|---|---|
| `inference.test.ts` | parse fixture v2; ค่า lesion FDI 45 (id/surface/probability 0.3152/20 จุด/centroid); ซี่ปกติ `caries: []` (27 ซี่); strip `confidence`/`bbox` ที่ฉีดเข้าไป; `surface: null` ผ่าน; **reject** เมื่อไม่มี `caries`, ใช้ `confidence` แทน `probability`, surface ไม่รู้จัก, probability > 1, mask < 3 จุด; idle/processing/fail/malformed |
| `analysisTypes.test.ts` | label/summary จาก fixture (29 ซี่, ผุ 2 ซี่, 2 surfaces); `caries[]` ส่งต่อครบ (FDI 26: 59 จุด, 0.5933); ซี่ปกติได้ `[]`; ไม่มี `confidence`; `surface: null` ผ่าน; colorKey; empty/zero/3 surfaces |
| `FindingsTable.test.tsx` | sort default (26, 45 แล้วเรียง FDI) / sort FDI ↑↓; ไม่มีคอลัมน์ Confidence (header เหลือ FDI + Caries surfaces, 2 cells ต่อแถว); empty state |
| `ToothDetailPanel.test.tsx` | surface probability null → "—" / เปอร์เซ็นต์; ไม่มี "Detection confidence"; ไม่มีตาราง lesion เมื่อ `caries: []`; ตาราง lesion หลายแถว; `surface: null` → "Unspecified" |
| `layerState.test.ts` | defaults (masks/caries/labels เปิด, axes ปิด); ไม่มี `boxes`; toggle ทีละ layer; toggle `caries` ไม่กระทบ `masks`; `setAll` |
| `overlays.test.ts` | ไม่ export `drawBoundingBox`; `drawCariesMasks` วาด polygon ตรงพิกัด + fill/stroke สี caries; หลาย lesion = หลาย path; ซี่ปกติไม่วาด; save/restore สมดุล; fill ของ tooth ≠ fill ของ lesion; halo ไม่เรียก `strokeRect` |
| `CanvasViewer.test.tsx` | มีปุ่ม Masks / Caries Mask / Axes / Labels และไม่มี Boxes; Caries Mask เปิดเป็น default; วาด lesion fill 2 ครั้ง (fixture) แล้วหยุดเมื่อปิด และกลับมาเมื่อเปิด; ปิดเฉพาะ Masks แล้ว lesion ยังวาด |

ตรวจ mutation แล้ว: เปลี่ยน `if (layers.caries)` เป็น `if (true)` ชั่วคราว เทสต์ toggle fail ตามคาด แล้วคืนโค้ดเดิม

### ผลรันล่าสุด (2026-11-14 21:04, `frontend/`)

| คำสั่ง | ผล |
|---|---|
| `npm run typecheck` | ผ่าน |
| `npm run lint` | ผ่าน |
| `npm test` | ผ่าน — 9 ไฟล์ / **53 tests** |
| `npm run build` | ผ่าน (375 modules; มีคำเตือนเดิมของ rollup เรื่อง annotation ใน `zod`) |
| `npm run format:check` | ผ่าน |

> ตัวเลขด้านบนคือผลรอบที่ 1 (53 tests) — ผลรอบที่ 2 อยู่ในหัวข้อ 5

---

## 5. รอบที่ 2 — Defensive guard ฝั่ง Frontend (ไม่แตะ backend)

เหตุผล: adapter ฝั่ง backend ไม่ตรวจ polygon ของ lesion (fallback เฉพาะ tooth mask) ถ้า ML ส่ง lesion ที่มี 0–2 จุด เดิม Zod `.min(3)` จะทำให้ผลทั้งงาน parse fail ทีมตกลงป้องกันที่ FE เองโดยไม่แก้ backend

| ไฟล์ | การแก้ |
|---|---|
| `src/domain/inference.ts` | `cariesLesionSchema.mask.data`: `z.array(pointSchema).min(3)` → `z.array(pointSchema)` (ยังบังคับให้ทุกจุดเป็น `[number, number]`). **tooth mask ยังต้อง ≥ 3 จุดเหมือนเดิม** (backend มี fallback เป็นสี่มุม bbox ให้แล้ว) |
| `src/components/CanvasViewer/overlays.ts` | `drawCariesMasks()`: เพิ่มค่า `MIN_POLYGON_POINTS = 3`; ข้าม (`continue`) lesion ที่ `mask.data.length < 3` ก่อนเรียก `tracePolygon` — lesion อื่นบนซี่เดียวกันยังวาดปกติ. ไม่ต้องแก้ `analysisTypes.ts` เพราะ view-model ส่ง `caries[]` ผ่านตามเดิม และตาราง "Caries lesions" ใน `ToothDetailPanel` ยังแสดงทุก lesion รวมตัวที่ polygon สั้น (ไม่ผูกกับ mask) |
| `src/fixtures/ml-result.raw.sample.json` | **ลบ** (`git rm`) — ไม่มีโค้ดหรือเทสต์ใช้แล้ว |
| `README.md` | แก้ส่วน fixtures/Mocking ที่อ้างถึงไฟล์ที่ลบ ให้ชี้ `result.sample.json` + `test-utils/v2Fixture.ts` |

เทสต์ที่เพิ่ม/เปลี่ยน:
- `inference.test.ts`: ลบเคสเดิมที่ "reject mask < 3 จุด"; เพิ่ม `it.each([0,1,2])` — payload ที่ lesion มี 0/1/2 จุด **ต้อง parse ผ่านทั้งก้อน** (29 ซี่ครบ, lesion ของ FDI 26 ไม่ถูกกระทบ); เพิ่มเคสที่จุดที่ไม่ใช่คู่ `[x, y]` ยังถูก reject
- `overlays.test.ts`: `it.each([0,1,2])` — `drawCariesMasks` ไม่ throw, ข้าม lesion สั้น และวาดเฉพาะ lesion ที่ถูกต้อง (1 path / 1 fill)
- `CanvasViewer.test.tsx`: render ทั้ง component ด้วย FDI 45 ที่ lesion เหลือ 2 จุด — ไม่ crash และทุก frame มี lesion fill เพียง 1 จุด (ของ FDI 26)
- ตรวจ mutation: ลบ guard ออกชั่วคราว เทสต์ใหม่ fail 3 เคส (กรณี 1, 2 จุด และเทสต์ CanvasViewer) แล้วคืนโค้ด (กรณี 0 จุดผ่านเพราะ `tracePolygon` คืน false กับ ring ว่างอยู่แล้ว)

ผลรันรอบที่ 2 (2026-11-14, `frontend/`): `typecheck` ผ่าน · `lint` ผ่าน · `test` ผ่าน 9 ไฟล์ / **61 tests** · `build` และ `format:check` ผลอยู่ในหัวข้อ 6



## 6. ผลตรวจก่อน commit (รอบที่ 2) และการ commit/push

| คำสั่ง | ผล |
|---|---|
| `npm run typecheck` | ผ่าน |
| `npm run lint` | ผ่าน |
| `npm test` | ผ่าน — 9 ไฟล์ / **61 tests** |
| `npm run build` | ผ่าน (375 modules) |
| `npm run format:check` | ผ่าน |

Commit: `feat(frontend): implement API v2 draft with caries lesion mask overlay and defensive polygon guard`
ขึ้น branch `feat/sprint_1_sem2_be` — stage เฉพาะ `frontend/` และไฟล์นี้ (ไม่รวม `ml-service/weights/*` และ `z-claude-md/proposal.md` ที่เป็น untracked เดิม) ไม่มีการแก้ `backend/`
ยังไม่ได้ทดสอบกับ backend/ML จริง และยังไม่ได้เปิดดูใน browser
