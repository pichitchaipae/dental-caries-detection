ช# ตรวจสอบ train–inference accuracy — 29 ตุลาคม 2026

## ข้อสรุป

ความต่างหลักมาจากการเปลี่ยนโจทย์และวิธีประเมิน: notebook RF แยกผิวฟันโดยรู้ตำแหน่งรอยผุจาก ROI แล้ว ส่วน serving ปัจจุบันต้องตรวจรอยผุจากภาพก่อนและใช้ geometry heuristic แยกผิวฟัน จึงนำ accuracy 81.66% มาเป็น baseline end-to-end ของระบบปัจจุบันโดยตรงไม่ได้

ตรวจพบ dependency drift และช่องโหว่ด้าน reproducibility จริง แต่การทดสอบครั้งนี้ยังไม่พบ preprocessing เสียอย่างรุนแรงใน pipeline ปัจจุบัน โมเดลที่ deploy เป็นไฟล์เดียวกับ best.pt และ validation ซ้ำได้ผลใกล้กับตอน train

การตรวจครั้งนี้เพิ่มเฉพาะสคริปต์วิเคราะห์และรายงาน ไม่เปลี่ยน serving code, thresholds, weights หรือ container ที่ให้บริการ และไม่ส่ง job ใหม่เข้า production DB

## 1. ยืนยันสภาพระบบจริง

- `/health`: ready=true, caries_enabled=true, artifacts ทั้งสาม verified, ไม่มี active job ขณะตรวจ
- SHA-256 ของ `detection.py`, `registry.py`, `orchestrator.py` ใน container ตรงกับ workspace
- checkpoint ที่ serve: `caries_crop_seg_20261028.pt` ตรงกับ `runs/caries_crop_seg/v1/weights/best.pt` ทุก byte
- SHA-256: `c0d4768fdbd90567feb28c5df34fa760321e79a08acbb47e11ba84a5c2c67d74`
- settings จริง: CPU, conf=0.25, caries NMS IoU=0.50, pano conf=0.25, Crop-Seg เปิด, PAD=20, suppress neighbours เปิด
- actual YOLO predict args ที่ตรวจได้: imgsz=640, rect=true, retina_masks=false; pano NMS IoU=0.70
- unit tests ด้าน detection, transform, crop, heuristic, XML และ output: **37 passed**

หลักฐาน: `ml-service/models/versions.py`, `ml-service/models/registry.py`, `ml-service/app/config.py` และ JSON ใน `outputs/` ที่ระบุท้ายรายงาน

## 2. P1 — RF baseline วัดคนละงานและไม่ได้เป็น held-out accuracy

ไฟล์: `reserch/phase2-1april/SP-Model-Optimized/rf-model-optimized-pipeline.ipynb`

1. Cell 18 อ่าน `caries_coordinates` จาก Week 3 mapping เพื่อสร้าง features ของรอยผุเทียบกับ tooth mask
2. `reserch/week3/dental_caries_analysis.py::calculate_caries_overlap()` สร้าง coordinates เหล่านี้จาก binary ROI mask ที่เตรียมไว้ ไม่ใช่จาก caries detector ที่กำลัง serve
3. Cell 20 ใช้ GroupShuffleSplit แยก train/test ตาม case ถูกต้อง แต่ `main()` ไม่ได้นำ test_dataframe ไปใช้คำนวณ final accuracy
4. Cells 26/28/36 ทำนายและ evaluate กลับบนทั้ง 500 cases รวม train cases
5. `match_case()` เดินเฉพาะ GT records; prediction บนฟันที่ไม่มี GT ไม่ถูกเพิ่มเป็น false positive ใน accuracy นี้
6. training ใช้ dictionary keyed ด้วย tooth_id ทำให้หลาย surface ของฟันซี่เดียวถูกเหลือ label สุดท้าย ส่วน inference ก็คืนหนึ่ง surface ต่อฟัน
7. `evaluate_version()` อ่าน relative path `PCA_Output_Production13F` แต่ writer ใช้ absolute OUTPUT_ROOT จึงเสี่ยงอ่านผลเก่าหรือคนละ directory ถ้าเปลี่ยน working directory

Stored notebook output: 1,979 records, accuracy **0.8166**, macro F1 **0.6311** เป็นผลที่บันทึกใน notebook ไม่ใช่ผลที่ retrain ใหม่ในการตรวจครั้งนี้ และไม่ใช่ held-out end-to-end caries detection accuracy

Serving ปัจจุบันใน `app/pipeline/orchestrator.py` import `findings_for_lesions` จาก `surface_heuristic.py` ไม่มีการเรียก RF หรือ CNN

## 3. P1 — โมเดลตรวจรอยผุปัจจุบันมีข้อจำกัดตั้งแต่ baseline

รัน YOLO validation ใหม่กับ **626 val crops / 354 lesion instances** โดยใช้ pinned checkpoint และ ML image ปัจจุบัน:

- Box precision 0.58843, recall 0.59322, mAP50 0.56326, mAP50–95 0.21217
- Mask precision **0.53728**, recall **0.50840**, mAP50 **0.48428**, mAP50–95 **0.15732**

ตัวเลขตรงกับรายงาน v2 เดิมที่ระบุ mask mAP50 ประมาณ 0.484 จึงไม่มีหลักฐานว่าประสิทธิภาพบนชุด crop เดิมตกอย่างมากหลัง deploy

Checkpoint metadata บันทึก Ultralytics **8.4.175**, mask mAP50=0.49127; container ปัจจุบันใช้ **8.4.191** และได้ 0.48428 ต่างกันประมาณ 0.70 percentage point ยังไม่ได้แยกทดลอง version, precision และ batching ทีละปัจจัย จึงยังระบุสาเหตุของส่วนต่างเล็กนี้ไม่ได้

หลักฐาน: `outputs/validation-20261029.json`

## 4. P1 — Distribution ของ evaluation crops ต่างจากการใช้ทั้ง panorama

`scripts/prepare_caries_crop_dataset.py` ใช้ negative sampling เพียง 3 ฟันต่อ case:

- ทั้ง dataset: positive 1,743 crops, negative 1,500 crops; positive ประมาณ **53.7%**
- ข้าม negative crops อีก **11,870** crops ตาม manifest
- val: positive 326 crops, negative 300 crops; มี 354 lesion instances ที่ถูกบันทึกเป็น labels
- serving ตรวจทุกฟันประมาณ 30 ซี่ต่อ case จึงเจอ negative/unannotated teeth มากกว่าชุด crop validation อย่างมาก
- val end-to-end ในรายงานเดิมมี GT 373 lesions มากกว่า 354 labels ใน saved val crops; การประเมิน crop จึงไม่ครอบคลุมทุก GT ที่ pipeline เต็มต้องค้นหา

นี่เป็น distribution shift ที่ยืนยันได้ แต่ยังไม่ได้ทำ ablation เพื่อพิสูจน์ว่า negative sampling เป็นสาเหตุใหญ่ที่สุดของ FP การเพิ่ม negatives/hard negatives เป็นการทดลองถัดไป ไม่ใช่ผลแก้ไขที่พิสูจน์แล้ว

ผล evaluation **เดิม** ของ 100 val cases ใน `ml-service/runs/caries_eval.json` ที่ conf=0.25, match mask IoU>=0.30:

- TP=246, FP=222, FN=127
- Precision=52.56%, recall=65.95%, F1=58.50%
- Surface accuracy 88.21% คิดเฉพาะ **246 lesions ที่ match แล้ว** ไม่รวม FN หรือ FP
- Surface ถูก 217 lesions; ถ้าบังคับถูกทั้ง lesion matching และ surface: TP=217, FP=251, FN=156, precision=46.37%, recall=58.18%, F1=51.61%

ตัวเลขชุดสุดท้ายคำนวณจาก confusion/counts ที่บันทึกไว้ ไม่ใช่ accuracy ใหม่บนทุกฟัน และไม่ใช่ผลรันครบ 100 cases ใหม่ในการตรวจนี้

ไฟล์เก่า `outputs/final_500_production_metrics.json` ใช้คำว่า Accuracy กับสูตร TP/(TP+FP+FN) ซึ่งเป็น Jaccard ของคู่ FDI+surface และใช้ pipeline/tuning เก่า จึงเทียบ 10.74% ในนั้นกับ RF accuracy 81.66% ไม่ได้

## 5. ทดสอบ inference ใหม่เพื่อแยก preprocessing ออกจาก detector errors

รัน pipeline จริงแบบ offline บน GPU: cases 10, 20, 42 และรัน CPU ซ้ำที่ case 10; ไม่ผ่าน public job API เพื่อไม่เปลี่ยนงานของผู้ใช้

- **Case 10:** Crop-Seg refine 29/29; saved val crops ทั้ง 5 ภาพตรงกับ crops ที่สร้างใหม่บน GPU ทุก pixel ตรวจเจอ FDI 26 distal ถูก, พลาด FDI 27 mesial และเกิด FP ที่ FDI 45 distal; TP/FP/FN=1/1/1 ที่ mask IoU>=0.30
- **Case 20:** refine 31/31; 8/10 saved crops ตรงทุก pixel อีกภาพต่างที่ขอบ suppression ประมาณ 0.231% ของ pixels และอีกภาพกว้างต่าง 1 pixel; TP/FP/FN=7/2/2 ที่ IoU>=0.30 และ 5/4/4 ที่ IoU>=0.50
- **Case 42:** refine 31/31; saved positive crops ของ FDI 47,46,15,35 มีอยู่และสร้าง crop ได้ แต่ไม่ตรวจเจอ GT ทั้ง 4 lesions; มี FP ที่ FDI 48 mesial; TP/FP/FN=0/1/4
- **CPU vs GPU case 10:** ได้สอง findings เดียวกันและ centroids ตรงกัน; confidence ต่างสูงสุดประมาณ 0.000215 ไม่มีการข้าม threshold; saved crop ต่างเล็กน้อยบริเวณ mask/suppression บางฟัน (<0.09% pixels ต่อ crop ที่ต่าง)

จึงยังไม่มีหลักฐานว่า device หรือ crop preprocessing เป็นสาเหตุหลักของผลตกในตัวอย่างเหล่านี้ ผลต่าง crop ขนาดเล็กเป็นสิ่งที่ควรเก็บใน tolerance ของ parity test ไม่ใช่อ้างว่า pixel parity สมบูรณ์ทุกภาพ

เอกสาร v2 เดิมสรุป FP ฟันหนึ่งกับ FN อีกฟันว่าเป็น adjacent-FDI swaps และโทษ Pano-Seg โดยตรง แต่กรณี 10 มี crop FDI 27 ตรงกับ dataset อยู่แล้ว และ 45 กับ 27 อยู่คนละ quadrant ข้อมูลนี้พิสูจน์ได้เพียง miss+false positive ยังไม่พิสูจน์การสลับ FDI

หลักฐาน: `outputs/inference-parity-gpu-20261029.json`, `outputs/inference-parity-cpu-20261029.json`

## 6. P2 — จุดคลาดเคลื่อนใน implementation และการรายงานที่ควรแก้

1. **Dependency ไม่ pin:** `requirements.txt` ใช้ `ultralytics>=8.3.0`; runtime .191 ต่างจาก checkpoint .175 ทำให้ rebuild ไม่ได้ environment เดิม ควร pin validated version และบันทึก package versions ในผลทุก run
2. **Device ไม่ถูกส่งไป YOLO:** `load_models_in_child(weights_dir, device)` ใช้ device กับ Detectron2 เท่านั้น ทั้ง YOLO ไม่มี `.to(device)` หรือ predict device explicit; เครื่องที่มองเห็น GPU อาจไม่ทำตาม ML_DEVICE ที่ตั้งไว้ ผล CPU/GPU sample ครั้งนี้ใกล้กัน แต่ contract ยังไม่ครบ
3. **YOLO predict args ไม่ explicit:** serving ไม่ส่ง imgsz/rect/retina_masks; caries manifest มี imgsz=640 แต่ code ไม่อ่าน field นี้ ปัจจุบัน default ตรง 640 แต่เปลี่ยน checkpoint/default อาจเกิด skew
4. **Detectron2 color format:** registry cfg.INPUT.FORMAT จริงเป็น BGR แต่ detection.py กลับช่องเป็น RGB เสมอ ต่างจาก DefaultPredictor ซึ่ง reverse เฉพาะ cfg=RGB ภาพที่ตรวจเป็น grayscale ทุกช่องเท่ากันจึงไม่กระทบกรณีนี้ แต่ comment และ parity claim ไม่ตรง config สำหรับภาพมีสี
5. **Pano resolution ใน heuristic validation ต่าง serving:** `validate_surface_heuristic.py` default imgsz=1024 แต่ serving ใช้ 640; ค่า agreement 79.7% จึงไม่ได้วัดจาก mask generation configuration เดียวกับ production
6. **Fallback อาจซ่อนความเสื่อม:** per-tooth Crop-Seg exceptions ถูกเปลี่ยนเป็น pano-mask fallback; /health ตรวจ checksum ไม่ทดสอบ model forward หรือ refinement rate ปัจจุบัน sample refine ครบ จึงไม่ใช่เหตุที่พบในรอบนี้
7. **Caries failure ยังให้ระบบ ready:** artifact caries missing/checksum_fail ทำให้ loader คืน None และไม่มี findings แต่ readiness ยังขึ้น true ได้โดยออกแบบ; result meta ยังใช้ชื่อจาก manifest แม้ model ไม่ได้ load ควรบันทึก actual loaded models และ degraded status ให้ชัด ปัจจุบันทั้งสาม artifacts verified
8. **Evaluation ไม่ใช้ runtime configuration ทั้งหมด:** eval script ใช้บาง defaults ของ run_stage1 และ surface rules ไม่รับ settings ทั้งชุด; ปรับ env แล้ว eval อาจไม่ตรง serving
9. **ไม่มี independent final test:** train ใช้ val เลือก checkpoint และ eval ใช้ val เดิมเลือก threshold อีกครั้ง ผลที่รายงานควรระบุว่า validation/tuning estimate แล้วเก็บ disjoint test cases เพิ่ม
10. **GT-first greedy matching:** `_match()` เลือกตามลำดับ GT ไม่ใช่ confidence-ordered AP matching และใช้ IoU 0.30; ตัวเลขนี้จึงไม่ใช่ YOLO mAP50 และเปลี่ยนได้หากเปลี่ยนลำดับ GT ในกรณีซ้อนกัน

## 7. ลำดับงานที่แนะนำ

1. กำหนด baseline แยกสามงาน: lesion detection, surface classification บน matched lesions, และ end-to-end lesion+surface พร้อม FP/FN ห้ามรวมเรียกว่า accuracy ค่าเดียว
2. ตรึง artifact + environment + preprocessing + predict configuration และเพิ่ม actual runtime provenance ในผล
3. ทำ validation โดยใช้ทุกฟันและ configuration เดียวกับ serving; แบ่ง tuning/test แยกตาม case และตรวจ patient IDs หากหนึ่งคนมีหลาย case
4. ตรวจ FN กับ FP crops โดยเฉพาะ cases 10/42, ทบทวน label completeness และทดลองเพิ่ม negatives/hard-negative mining โดยรักษา test set เดิมไว้
5. หากต้องการใช้ RF ต่อ ต้องประเมิน RF ด้วย **predicted lesion masks** จาก upstream detector ตามสภาพใช้งานจริง ไม่ใช้ GT ROI geometry มาสรุป end-to-end performance

## ขอบเขตและการรันซ้ำ

- อ่าน training notebook, model manifest, data-generation/training/evaluation scripts, backend image store/result adapter, frontend findings และ runtime code
- รัน validation ใหม่ครบ 626 saved val crops
- รัน end-to-end ใหม่ 3 validation cases บน GPU และ 1 case บน CPU
- ใช้ผล 100-case end-to-end จาก artifact ที่มีอยู่; ไม่ได้รันทั้ง 100 หรือ 500 cases ใหม่ ไม่ได้ retrain RF/YOLO และไม่ได้แยก ablation แต่ละ causal factor
- อาการในภาพของผู้ใช้รายใดรายหนึ่งยังต้องเทียบกับ input เดียวกันก่อนอ้างสาเหตุเฉพาะภาพนั้น

สคริปต์ที่เพิ่ม: `ml-service/scripts/audit_inference_parity.py` รองรับ images root, dataset, weights, case IDs, device และ output JSON; ตัวอย่างอยู่ใน docstring
