# DEPLOYMENT MANIFEST — Caries Surface CNN (EfficientNet-B0, image domain)

**สถานะ: STAGING ONLY — ยังไม่รับรองทางคลินิก / ไม่ใช่ external validation**

- วันที่: 2026-10-18
- บริการ: `deploy_effnet_service.py` (FastAPI, `/health`, `/model/info`, `POST /v1/surface/predict`)
- ชุดน้ำหนัก: `outputs/best_pca_s42_f{0..4}.pt` (5 fold, ensemble = ค่าเฉลี่ย per-fold sigmoid)
- Config ล็อก: `ablation=pca`, `backbone=efficientnet_b0`, `loss=focal`, `img_size=384`, `epochs=14`,
  `cv=stratified-groupkfold`, `n_folds=5`, `exclude=@peeked(100 cases)`, `selection=macro_ap`,
  `recall_target=0.80`, `seed=42` (+ seed 43 รอผล) — ตาม `outputs/PROTOCOL_LOCK.md`

> ไฟล์นี้ผูก "โค้ด + น้ำหนัก + การตัดสินใจ" ให้ตรวจสอบย้อนกลับได้
> **ห้ามแก้ค่าในไฟล์นี้ด้วยมือ** — ต้องคัดลอกจากผลรัน/`/health` จริงเท่านั้น

---

## 0. Code revision (ซอร์สโค้ดที่ผลิต artifacts ชุดนี้)

| รายการ | ค่า |
|---|---|
| commit | `631c3a36` (`git rev-parse HEAD` สำหรับ hash เต็ม) |
| ข้อความ | `feat(caries-detection-model): image-domain CNN surface classifier + CPU staging service` |
| ขอบเขต | 16 ไฟล์ / +5,466 บรรทัด — รวม `.gitignore`, `PROTOCOL_LOCK.md`, `deploy/DEPLOYMENT_MANIFEST.md` |
| repo / branch | `D:\Mahidol University\SP` (branch `main`) |
| คำสั่งที่ใช้ผลิต artifacts | `run_stage2_matrix.ps1` (คิว 4 run: `pca42` → `off42` → `pca43` → `off43`) · `run_calibrate_cpu.py` (บังคับ CPU) · `deploy/run_staging.ps1` |
| **ไม่**รวมใน commit | `outputs/**` (weights, calibration, OOF, evidence) — ผูกด้วย SHA256 ในตาราง §1.1-§1.2 แทน |

> `631c3a36` เกิดจาก `git commit --amend -F` ของ `8d6cef7` เพื่อแก้คำผิดในข้อความ commit เท่านั้น
> (ไม่มีไฟล์ใดถูก stage ตอน amend → tree ของทั้งสอง commit เหมือนกันทุกไบต์) — **เวลาอ้างในรายงานให้ใช้ `631c3a36`**

---

## 1. Artifacts และ SHA256

### 1.1 Calibration / evidence (ยืนยันแล้ว)

| ไฟล์ | SHA256 | ที่มา |
|---|---|---|
| `outputs/calibration_pca.json` | `D22036F398604043482C862019F1607325C0627F959C100F6E596EC8251B9DF1` | `Get-FileHash` (ผู้ใช้รัน) |
| `outputs/oof_pca_stratified-groupkfold_seed42_exc100.csv` | `D759A6369F8318BFE6802BD57AB9AE121E4D923D701D23F0058FA93740DADD32` | `Get-FileHash` (ผู้ใช้รัน) |
| `outputs/oof_report_oof_pca_stratified-groupkfold_seed42_exc100.json` | `072D59974A348AE900975E3B0CA75D19C6BC2F767EFE5AB9E9CCAA02EA0099F9` | `Get-FileHash` (ผู้ใช้รัน) |
| `outputs/deploy/health_8013.json` (runtime snapshot; `captured_at` อยู่ในไฟล์) | `5FBE8CDB047CF0AC3D94AA45445CA28EFE85F634FAB9A9462614D4FB1495074F` | `Invoke-RestMethod /health` + `Get-FileHash` (2026-10-19) |

### 1.2 Checkpoint น้ำหนัก 5 fold (weights ที่ deploy จริง)

| ไฟล์ | SHA256 |
|---|---|
| `outputs/best_pca_s42_f0.pt` | `266AD0172677271A5E2B9686808CD3F42B11377CF735540937FD19EA38DFC57A` |
| `outputs/best_pca_s42_f1.pt` | `65BAB185DA59C80B08C865B23A6A1A77AA65BCD7292096C801E5E6E0290281EF` |
| `outputs/best_pca_s42_f2.pt` | `BEEE3E30A7337760D8C553C32D861D572DEE632C16C43832345C62C34B870ED1` |
| `outputs/best_pca_s42_f3.pt` | `1BF358A027AA8FCEBAEE297F7E30B15B677E2F84264340D09C9D0AE5D807A83B` |
| `outputs/best_pca_s42_f4.pt` | `4E1D6838EE4259D666CB00153FF7141CDB2E6193F08D681ADF730B02DD2F6837` |

ที่มา: `Get-FileHash` โดยผู้ใช้ (2026-10-18) ตรงกับ sha256 ที่ `/health` รายงานตอนโหลด 5 fold

> **ไม่นับเป็น weights ของ release:** `outputs/best_pca_s42_f3_last.pt`, `best_pca_s42_f4_last.pt`
> (`C5080407…E5CA`, `23FDB6DE…5ADD`) เป็น **snapshot ต่อ epoch สำหรับ resume** (`pipeline.py:1439` เขียน
> `{prefix}_last.pt` ทุก epoch — หลักฐานว่าไม่ใช่ของเฉพาะ fold ที่ resume: run `off42` ที่เทรนรวดเดียว
> มีไฟล์นี้ **ครบทั้ง 5 fold**) · สาเหตุที่ `pca42` เหลือเฉพาะ f3/f4 **ยังไม่ได้ตรวจยืนยัน** (ไม่กระทบการให้บริการ)
> — `/health` และ `glob("best_pca_s42_f[0-9].pt")` ยืนยันว่าโหลด 5 checkpoint ตามเลข fold เท่านั้น ไม่หยิบ `_last`
>
> คำสั่งแจกแจง weights ของ release (กรอง scratch ออก):
>
> ```powershell
> $root = "D:\Mahidol University\SP\caries-detection-model"
> Get-ChildItem "$root\outputs\best_pca_s42_f*.pt" |
>   Where-Object { $_.Name -notlike "*_last*" } |
>   Sort-Object Name |
>   ForEach-Object { "$($_.Name)`t$((Get-FileHash $_.FullName -Algorithm SHA256).Hash)" }
> ```

ทางเลือกที่ตรงกับสิ่งที่บริการโหลดจริง (บันทึกเป็นหลักฐานไว้ด้วย):

```powershell
Invoke-RestMethod http://127.0.0.1:8013/health |
  ConvertTo-Json -Depth 8 |
  Set-Content "$root\outputs\deploy\health_8013.json" -Encoding utf8
```

### 1.3 Environment (รอกรอก)

| รายการ | ค่า |
|---|---|
| Python env | `D:\Users\jaopi\anaconda3\envs\sp_project\python.exe` |
| torch | `2.5.1` |
| torchvision | `0.20.1` |
| scikit-learn | `1.7.2` |
| OS | Windows (win32) |

ที่มา: `pip show torch torchvision scikit-learn | Select-String "^(Name|Version)"` (ผู้ใช้รัน 2026-10-18)

หมายเหตุ: การเทรนใช้ GPU แต่ **บริการรัน CPU เป็นค่าเริ่มต้น** (สมมติฐานแวดล้อม: torch build ที่ติดตั้งอาจมี CUDA — ถ้าต้องการระบุให้ครบ: `python -c "import torch;print(torch.__version__, torch.version.cuda)"`)

---

## 2. สัญญาการใช้งาน (input/output contract)

**Input** — `POST /v1/surface/predict` (`multipart/form-data`)

| ฟิลด์ | ชนิด | ความหมาย |
|---|---|---|
| `crops` | ไฟล์ PNG (grayscale) | crop ฟัน 1 ซี่ ต่อ 1 ไฟล์ (ต้องเป็น crop ชุดเดียวกับตอนเทรน) |
| `masks` | ไฟล์ PNG (ขาว/ดำ) | mask ของซี่นั้น ขนาดเท่า crop |
| `fdis` | string | เลข FDI คั่นด้วยจุลภาค เรียงตรงกับไฟล์ |
| `fold` | int (optional) | บังคับใช้โมเดล fold เดียว (ใช้เพื่อทดสอบ parity เท่านั้น) |

**Output** — ต่อซี่: `fdi`, `surfaces.{Distal,Mesial,Occlusal}.{probability, threshold, positive}`, `probabilities[3]`, และ `model` (checkpoint list, `calibration`, `staging_only`)

- **เงื่อนไขสำคัญ:** `crops`/`masks`/`fdis` ต้องมีจำนวนเท่ากัน ไม่งั้นคืน **HTTP 422** (ทดสอบแล้ว)
- ความหมายของ `positive` = `probability >= threshold` — ตรวจสอบสดแล้ว 18/18 ครั้ง

---

## 3. ชั้นตัดสินใจ (Decision layer)

ลำดับการคำนวณ (เหมือนกันทั้งตอน calibrate และตอน serve — ตรวจในโค้ดแล้ว):

```
probs = mean_fold( sigmoid(logit_fold) )        # ensemble เฉลี่ยจากความน่าจะเป็น
z     = logit(clip(probs, 1e-6, 1-1e-6))
probs = sigmoid(z / T)                          # temperature scaling
positive = probs >= threshold
```

| Surface | threshold (deployment) | อ้างอิง溫度 T |
|---|---|---|
| Distal | `0.2119` | ดู `calibration_pca.json` → `temperature.Distal` |
| Mesial | `0.3094` | ดู `calibration_pca.json` → `temperature.Mesial` |
| Occlusal | `0.0587` | ดู `calibration_pca.json` → `temperature.Occlusal` |

ที่มา: `pipeline.py calibrate --calib "@peeked"` บน **100 เคสที่ถูก peek ไปแล้ว (3,019 ซี่)** ซึ่งทุก fold
กันออกจาก training → ค่าที่ได้เป็น **diagnostic เท่านั้น ห้ามอ้างเป็น performance**
(Arm: ปรับจาก 0.816/0.769/0.785 → 0.2119/0.3094/0.0587 หลังแก้บั๊ก calibrate-set; ไฟล์ที่ผิดเก็บไว้ที่
`outputs/calibration_pca_INVALID_insample.json` เป็นหลักฐานเหตุการณ์)

> **ห้ามสับสนกับ `/health`:** ฟิลด์ `checkpoints[].thresholds` ใน `/health` คือ threshold ของ **tune-half
> แต่ละ fold** (metadata ค่ากรอบ 0.05 เช่น `0.4 / 0.55 / 0.65`) — **ไม่ใช่** threshold ที่ใช้ตัดสินใจ
> ตัวที่ใช้จริงคือ `calibration.threshold` ในตารางข้างบนเท่านั้น

**Rollback lever (ปลอดภัย):** ลบ/ย้าย `calibration_pca.json` ออกแล้วรีสตาร์ต → บริการกลับเป็น
**probability-only** (`threshold=null`, `positive=null`) โดยไม่ล้ม — ตรวจในโค้ดบรรทัด 230-232 แล้ว

---

## 4. หลักฐานการตรวจสอบ 3 ระดับ (ก่อนถือว่า deploy ได้)

| # | การตรวจ | วิธี | ผล | เกณฑ์ | หลักฐาน |
|---|---|---|---|---|---|
| 1 | Preprocessing/inference parity | `deploy/smoke_test_effnet.py --fold 0` บน instance **ไม่ calibrated** (พอร์ต 8014) | metric max\|diff\| = **5.926e-04**, per-tooth = **2.980e-08**, 1,225 ซี่/41 เคส, จำนวนไม่ตรง → 422 | < 2e-3 / < 1e-5 | `outputs/deploy/smoke_f0_20261018T131524Z.json` |
| 2 | Readiness/artifacts | `GET /health`, `/model/info` | ready, 5 folds, calibration loaded, `errors=[]`, sha256 ตรง manifest | — | `outputs/deploy/health_8013.json` (sha256 `5FBE8CDB…074F`) |
| 3 | Decision layer | ยิง 6 ซี่จริงผ่าน HTTP เทียบกับ CSV | `positive == (prob >= thr)` ถูกทั้ง **18/18** (รันซ้ำ 2 ครั้ง ได้ค่าเดิมทุกหลัก) | 100% | ผลรันผ่าน `check_8013.py` |

หมายเหตุ: ค่า 5.926e-04 **ซ้ำเดิมทุกหลัก** ในการรันซ้ำ → reproducible และเป็นสัญญาณรบกวน
CPU-vs-CUDA float32 ไม่ใช่ความคลาดเคลื่อนของโค้ด (tolerance ข้ามอุปกรณ์ ~2e-3)

> **กับดักที่ต้องรู้:** `deploy/smoke_test_effnet.py` **จะ FAIL** ถ้ารันกับ instance ที่โหลด calibration
> (เพราะเทียบ "ความน่าจะเป็นที่ถูก temperature แล้ว" กับ threshold ที่บันทึกจากตอนไม่ calibrate)
> → ต้องรัน parity บน instance ที่ **ไม่** ใส่ `-Calibration` เท่านั้น

---

## 5. ผลลัพธ์ที่มีอยู่จริง (OOF, pooled 5 folds — 6,012 ซี่ / 201 เคส)

| Surface | AP (mean±sd) | Recall [Wilson 95%] | Precision | Brier |
|---|---|---|---|---|
| Distal | 0.629 ± 0.075 | 0.813 [0.770–0.849] | 0.338 | 0.107 |
| Mesial | 0.552 ± 0.069 | **0.771** [ต่ำกว่าเป้า 0.80] | 0.261 | 0.098 |
| Occlusal | 0.604 ± 0.086 | 0.809 | 0.196 | 0.061 |
| ANY (มีรอยอย่างน้อย 1 ด้าน) | 0.705 ± 0.046 | 0.883 | 0.380 | 0.146 |

ที่มา: `outputs/oof_report_oof_pca_stratified-groupkfold_seed42_exc100.{json,md,png}`
(threshold ต่อ fold มาจาก **tune half ของ fold นั้น** — nested, ไม่ใช้ test)

---

## 6. ข้อจำกัดที่ต้องประกาศ (ห้ามละ)

1. **ไม่ใช่การรับรองทางคลินิก** — ไม่มี external validation, ไม่มี expert κ (ground truth มาจาก annotation ปัจจุบัน)
2. **Mesial recall 0.771 < เป้า 0.80** ที่ threshold แบบ nested — รายงานตามจริง ไม่ re-tune บน test
3. **Occlusal threshold 0.0587 เปราะ** — อยู่ใกล้กลุ่มความน่าจะเป็นส่วนใหญ่ (ตัวอย่างจริง: 0.0546 vs 0.0587)
   → คำตัดสินพลิกง่ายเมื่อข้ามเครื่อง/เครื่องสแกน; ทางเลือกระยะยาว: คืนค่า probability อย่างเดียว
   หรือยอมลด recall เพื่อ threshold ที่นิ่งกว่า
4. **Ensemble ในชุด OOF/smoke ไม่ใช่ภาพ deployment** — ในการทดสอบ 4 ใน 5 โมเดลเคยเทรนบนซี่นั้น
   (in-sample) → ห้ามอ่าน precision/recall จาก smoke test; ใช้ `fold` เมื่อต้องเทียบกับ OOF
5. **ยังไม่ทดสอบ end-to-end จากภาพพาโนดิบ** — บริการนี้รับ crop+mask; เส้นทาง
   pano → ตรวจหา/แบ่งซี่ → crop → predict ยังไม่เคยรันต่อกันจริง
6. **การเปรียบเทียบ arm (pca vs off, 2 seeds) รันแล้ว** (`compare-oof`, k=10, margin 0.05):
   Distal ΔAP **+0.029** (non-inferior; bootstrap 95% [+0.003, +0.054]) · Mesial **+0.031** (non-inferior; [−0.001, +0.065]) ·
   Occlusal **+0.021** (**inconclusive**; corrected-t [−0.104, +0.146]) — **ไม่มีด้านใดผ่านเกณฑ์ superior**
   (corrected-t lower bound > 0) → ห้ามสรุปว่า `pca` ดีกว่า `off`; อ้างได้เพียง "ไม่ด้อยกว่าที่ margin 5% ของ AP"
   ใน Distal/Mesial และ "สรุปไม่ได้" ใน Occlusal
7. **ความแปรปรวนระหว่าง seed มีขนาดใหญ่** (pca Distal AP 0.629 ที่ seed42 vs 0.690 ที่ seed43; Occlusal 0.604 vs 0.565)
   → ตัวเลข performance ต้องอ่านคู่กับ SD ต่อ fold และต้องระบุ seed ที่อ้างเสมอ (reference = seed 42 ตาม protocol;
   seed 43 เป็น replication — ห้ามเลือก seed ที่ตัวเลขดีกว่า)

---

## 7. วิธีรัน / หยุด / ย้อนกลับ

**รัน (staging, CPU):**
```powershell
powershell -ExecutionPolicy Bypass -File "D:\Mahidol University\SP\caries-detection-model\deploy\run_staging.ps1" `
  -Port 8013 -Folds "0,1,2,3,4" -Calibration outputs\calibration_pca.json
```

**หยุด:** ในหน้าต่างที่รันอยู่ กด `Ctrl+C` (หรือ `Stop-Process -Id <PID>`)

**ย้อนกลับทั้งหมด:** งานนี้เป็นแบบ **additive** — ไม่แก้เส้นทาง RF เดิม (`dental-caries-detection/ml-service`)
ไฟล์ที่เพิ่มมีเพียง: `deploy_effnet_service.py`, `deploy/`, `pipeline.py`, `tests/`, `run_stage2_*.{ps1,cmd}`,
`PIPELINE_DOCUMENTATION_TH_EN.md` → ลบ/ย้ายไฟล์เหล่านี้ = กลับสู่สภาพเดิม, ไม่มี migration ใด ๆ

---

## 8. ลายเซ็นผู้รับผิดชอบ

| บทบาท | ชื่อ | วันที่ | หมายเหตุ |
|---|---|---|---|
| ผู้พัฒนา/ผู้ตรวจสอบ | `<PENDING>` | 2026-10-18 | staging เท่านั้น |
