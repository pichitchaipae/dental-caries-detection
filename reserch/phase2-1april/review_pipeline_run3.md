# รายงานวิเคราะห์ทางเทคนิค: pipeline_run3_final.py  
## Dental Caries Surface Classification — Run 3

---

## 1. ภาพรวมของ Pipeline

`pipeline_run3_final.py` เป็น pipeline สำหรับจำแนกผิวฟันที่มีรอยผุ (caries surface classification) บนภาพ X-ray ทางทันตกรรมแบบ panoramic จำนวน 500 ราย โดยจำแนกออกเป็น 4 คลาส ได้แก่ **Occlusal**, **Mesial**, **Distal**, และ **Other** ซึ่งคลาส Other ถูกออกแบบให้เป็น fallback class สำหรับกรณีที่ไม่สามารถจำแนกได้ชัดเจน

Pipeline แบ่งการทำงานออกเป็น 6 ขั้นตอนหลักดังนี้:

```
create_ml_dataset()  →  train_classify_ml()  →  classify_ml()
     ↓                        ↓                      ↓
feature extraction       RF training         Smart Fallback
     ↓
process_case_ml()  →  evaluate_version()  →  plot/report
```

ผลลัพธ์ที่ได้จาก Run 3 คือ Accuracy 0.82–0.83 โดยมี per-class recall สำหรับ Occlusal อยู่ที่ 0.84, Mesial 0.80, และ Distal 0.84 ซึ่งบรรลุเป้าหมายที่ตั้งไว้ที่ recall > 0.80 สำหรับ 3 คลาสหลัก

---

## 2. เทคนิคที่ 1 — PCA-Based Tooth Orientation Alignment

### 2.1 หลักการทำงาน

ฟังก์ชัน `perform_pca()` รับพิกัด pixel ของ mask ฟัน (tooth mask) แล้วใช้ **Principal Component Analysis (PCA)** เพื่อหาแกนหลักของรูปร่างฟัน จากนั้นหมุนพิกัดทั้งหมดให้แกนยาวของฟันตั้งฉากกับแกน X (upright position) ก่อนนำไปคำนวณ feature ต่อไป

กระบวนการดำเนินตาม **4 กฎ** ที่กำหนดไว้อย่างชัดเจน:

1. **กฎที่ 1 — แยกแกนแนวตั้งออกจากแกนแนวนอน**: eigenvector ที่มีค่า |Y| สูงกว่าถูกกำหนดเป็นแกนแนวตั้ง (vertical axis) ส่วนอีกแกนเป็น horizontal axis

2. **กฎที่ 2 — กำหนดทิศทางแกนตั้งตามขากรรไกร**: ฟันบน (quadrant 1–2) กำหนดให้แกนตั้งชี้ลง (Y เพิ่มขึ้น) ส่วนฟันล่าง (quadrant 3–4) ชี้ขึ้น เพื่อให้ส่วน crown อยู่ในทิศทางเดียวกันเสมอ

3. **กฎที่ 3 — กำหนดทิศทางแกนนอนตาม quadrant**: quadrant 1 และ 4 ให้แกนนอนชี้ไปทางขวา ส่วน quadrant 2 และ 3 ชี้ไปทางซ้าย เพื่อรองรับ FDI notation ที่สมมาตรกัน

4. **กฎที่ 4 — Clamp extreme rotation**: หากมุมหมุนเกิน `MAX_TILT_DEG = 45°` ให้ reset เป็น 0 เพื่อป้องกันการหมุนที่ผิดพลาดจาก outlier pixel

```python
_, eigvecs = cv2.PCACompute(centered, mean=None)
# ... (4 rule logic) ...
rotation_angle = target_angle - angle_from_x
if abs(math.degrees(rotation_angle)) > MAX_TILT_DEG:
    rotation_angle = 0.0  # กฎที่ 4
```

### 2.2 เหตุผลที่เลือกใช้เทคนิคนี้

ในภาพ X-ray แบบ panoramic ฟันแต่ละซี่จะมีมุมเอียงที่แตกต่างกันตามธรรมชาติของกายวิภาค ประกอบกับความแตกต่างในการวางหัวผู้ป่วย หากไม่มีการปรับ orientation ก่อน feature ที่วัดตำแหน่ง X ของ caries จะให้ค่าที่ไม่สม่ำเสมอสำหรับฟันซี่เดียวกันในผู้ป่วยต่างคน PCA ช่วยสร้าง **canonical coordinate system** ที่ independent กับมุมในภาพดิบ ทำให้ feature ที่สกัดมาสะท้อน "ตำแหน่งบนฟัน" อย่างแท้จริง ไม่ใช่ตำแหน่งในภาพ X-ray

การใช้ `cv2.PCACompute` ยังมีข้อดีด้านประสิทธิภาพ เนื่องจากเป็น C++ implementation ที่เร็วกว่า `numpy.linalg.eig` สำหรับ array ขนาดเล็ก และรองรับ float32 โดยตรงซึ่งลดการแปลง dtype ที่ไม่จำเป็น

### 2.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

**Oriented Bounding Box (OBB)** ด้วย `cv2.minAreaRect()` เป็นทางเลือกที่ง่ายกว่า แต่ OBB ไม่รับประกันทิศทางที่สอดคล้องกัน (เช่น ไม่รู้ว่าแกนยาวของ OBB ชี้ขึ้นหรือลง) จึงยังต้องใช้ logic การแก้ทิศทางเพิ่มเติมอยู่ดี

**Anatomical Landmark Detection** โดยใช้ deep learning model แยกต่างหากเพื่อระบุจุด apex, cusp, และ cervical line จะให้ความแม่นยำสูงกว่า PCA มาก แต่ต้องการ annotation เพิ่มเติมและ model อีกตัวที่ต้องฝึกแยกต่างหาก

สำหรับ dataset ขนาด 500 รายที่ไม่มี landmark annotation พร้อม การใช้ PCA ถือว่าสมเหตุสมผล เนื่องจากให้ผลดีโดยไม่ต้องใช้ resource เพิ่ม

---

## 3. เทคนิคที่ 2 — DBSCAN Clustering สำหรับแยก Lesion

### 3.1 หลักการทำงาน

ฟังก์ชัน `split_caries_into_lesions()` รับพิกัด pixel ของ caries ทั้งหมดในฟันหนึ่งซี่ แล้วใช้ **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** เพื่อแยกกลุ่ม pixel ที่เชื่อมถึงกันออกเป็น lesion แต่ละจุดอย่างอิสระ

```python
clustering = DBSCAN(eps=2.0, min_samples=1).fit(pts)
# กรองขนาดขั้นต่ำหลัง clustering
lesion_pts = pts[labels == lbl]
if len(lesion_pts) >= min_cluster:
    lesions.append(lesion_pts)
```

พารามิเตอร์ที่เลือกใช้มีเหตุผลดังนี้:

- **`eps=2.0`**: บน pixel grid ระยะห่าง 2 pixel ครอบคลุม pixel เพื่อนบ้านแบบ 4-connected และ 8-connected ได้ทั้งหมด ทำให้ pixel ที่ติดกันถูกรวมเป็น cluster เดียวกัน
- **`min_samples=1`**: ใช้เพื่อให้ทุก pixel สามารถเป็น core point ได้ ซึ่งมีผลให้ไม่มี noise point (-1) เกิดขึ้น จากนั้นกรองด้วย `MIN_CLUSTER_SIZE = 15` pixel แยกต่างหากแทน แนวทางนี้แก้ปัญหาที่ใช้ `min_samples=15` โดยตรงซึ่งทำให้แต่ละ pixel ไม่สามารถหา 15 เพื่อนบ้านบน pixel grid ได้

### 3.2 เหตุผลที่เลือกใช้เทคนิคนี้

ฟันหนึ่งซี่อาจมีรอยผุมากกว่าหนึ่งจุด เช่น มี caries บนผิว Mesial และ Occlusal พร้อมกัน หากนำ pixel ทั้งหมดมาคำนวณ centroid รวมกัน ตำแหน่งที่ได้จะอยู่กลางระหว่าง lesion ทั้งสอง ซึ่งไม่ตรงกับผิวใดเลย การแยก lesion ด้วย clustering ก่อนจึงเป็นสิ่งจำเป็นเพื่อให้ feature ของแต่ละ lesion สะท้อนตำแหน่งที่แท้จริง

DBSCAN ถูกเลือกเหนือ algorithm อื่นเพราะ:
- ไม่ต้องกำหนดจำนวน cluster ล่วงหน้า (ต่างจาก k-means)
- สามารถค้นหา cluster ที่มีรูปร่างไม่สม่ำเสมอได้ (caries มีรูปร่างหลากหลาย)
- รองรับข้อมูล 2D coordinate ได้โดยตรงโดยไม่ต้องแปลงเป็น binary image

### 3.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

**Connected Component Analysis** (CCA) ด้วย `cv2.connectedComponentsWithStats()` เป็นทางเลือกที่มีประสิทธิภาพสูงกว่า DBSCAN มากสำหรับ binary pixel data โดยทำงานใน O(N) เทียบกับ DBSCAN ที่มีความซับซ้อน O(N log N) ถึง O(N²) ข้อจำกัดคือ CCA ต้องการ binary image เป็น input ซึ่งต้องแปลง coordinate list เป็น image matrix ก่อน แต่สำหรับ dataset ขนาด 500 รายซึ่งไม่ใช่ real-time application ความแตกต่างด้านความเร็วนี้ไม่ส่งผลกระทบอย่างมีนัยสำคัญ

---

## 4. เทคนิคที่ 3 — 13-Dimensional Geometric Feature Extraction

### 4.1 หลักการทำงาน

ฟังก์ชัน `_extract_ml_feature_dict()` สกัด feature 13 มิติจากคู่ (tooth, lesion) หลังจากที่ coordinate ทั้งหมดถูก PCA-align แล้ว โดย feature ทุกตัวเป็น **relative coordinate** ที่ normalized ด้วยขนาดของฟัน ทำให้ scale-invariant ต่อขนาดฟันและ resolution ของภาพ

```python
x_rel = np.clip((caries_rot[:, 0] - bbox_x) / w, 0.0, 1.0)
y_rel = np.clip((caries_rot[:, 1] - bbox_y) / h, 0.0, 1.0)
```

| Feature | คำอธิบาย | เหตุผลที่เกี่ยวข้อง |
|---|---|---|
| `is_upper` | ฟันบน (1) หรือฟันล่าง (0) | ทิศทาง Mesial/Distal สลับกันระหว่างขากรรไกร |
| `x_mean` | ค่าเฉลี่ย X ของ caries (0–1) | ตัวบ่งชี้หลักว่า caries อยู่ฝั่งไหนของฟัน |
| `y_mean` | ค่าเฉลี่ย Y ของ caries (0–1) | ความสูงของ caries บนฟัน |
| `x_std` | ส่วนเบี่ยงเบนมาตรฐาน X | ความกว้างแนวนอนของ caries |
| `y_std` | ส่วนเบี่ยงเบนมาตรฐาน Y | ความสูงของ caries |
| `x_min` | ตำแหน่ง X ซ้ายสุดของ caries | ขอบซ้ายของ caries |
| `x_max` | ตำแหน่ง X ขวาสุดของ caries | ขอบขวาของ caries |
| `y_min` | ตำแหน่ง Y บนสุดของ caries | ว่า caries เริ่มจากส่วน crown หรือไม่ |
| `x_range` | x_max − x_min | ความกว้างของ bounding box แนวนอน |
| `y_range` | y_max − y_min | ความสูงของ bounding box แนวตั้ง |
| `x_centroid_dist` | \|x_mean − 0.5\| | ระยะห่างจากกลางฟันในแนวนอน |
| `aspect_ratio` | w / h ของฟัน | บ่งบอก morphology ของฟัน (กว้าง/แคบ) |
| `coverage` | \|caries pixels\| / \|tooth pixels\| | ขนาดสัมพัทธ์ของ caries เทียบกับฟัน |

### 4.2 เหตุผลที่เลือกใช้เทคนิคนี้

Feature set นี้ออกแบบมาเพื่อตอบคำถาม "caries อยู่ที่ไหนในฟัน?" อย่างเป็น geometry-centric โดยตรง ข้อดีที่ชัดเจนคือ:

**Interpretability**: feature แต่ละตัวมีความหมายทางกายวิภาคที่ชัดเจน เช่น `x_mean` ต่ำหมายความว่า caries อยู่ฝั่ง Distal ส่วน `x_centroid_dist` สูงหมายความว่า caries ไม่ได้อยู่กลางฟัน (น่าจะเป็น Mesial หรือ Distal)

**Scale invariance**: การ normalize ด้วยขนาดฟัน (w, h) ทำให้ feature ไม่ขึ้นกับขนาดฟันจริง ซึ่งแตกต่างกันมากระหว่าง molar (ใหญ่) กับ incisor (เล็ก)

**Compatibility กับ small dataset**: สำหรับ 500 รายซึ่งหลังแยก lesion แล้วอาจได้ training samples ประมาณ 1,000–3,000 samples feature engineering แบบ hand-crafted ทำงานได้ดีกว่าการปล่อยให้ neural network เรียนรู้ representation เองจาก dataset ขนาดเล็กเช่นนี้

### 4.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

**Zernike Moments** หรือ **Hu Moments** สามารถเข้ารหัสรูปร่าง (shape) ของ caries region ได้ครบถ้วนกว่า feature ปัจจุบัน ซึ่งวัดเพียง bounding statistics เท่านั้น อย่างไรก็ตาม moment-based feature มีความซับซ้อนด้านการตีความและ sensitive ต่อ noise มากกว่า

**Deep Learning Feature Extraction** (CNN backbone เช่น ResNet) บน patch ของ caries ที่ crop มาจาก image จะสามารถเรียนรู้ representation ที่ซับซ้อนกว่าได้มาก แต่ต้องการ dataset ขนาดใหญ่กว่าและ labeling ในระดับ pixel อย่างละเอียด

---

## 5. เทคนิคที่ 4 — Random Forest Classifier

### 5.1 หลักการทำงาน

ฟังก์ชัน `train_classify_ml()` ฝึก `RandomForestClassifier` ด้วยการตั้งค่าดังนี้:

```python
model = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=200,
    random_state=42,
)
```

#### 5.1.1 GroupShuffleSplit by case_id

```python
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(
    feature_dataframe, groups=feature_dataframe["case_id"]
))
```

`GroupShuffleSplit` รับประกันว่าข้อมูล (lesion samples) จากผู้ป่วยรายเดียวกันจะไม่ถูกแบ่งคร่อมระหว่าง training set และ test set ฟังก์ชันนี้จำเป็นอย่างยิ่งสำหรับ medical imaging เนื่องจากผู้ป่วยหนึ่งรายอาจมีหลายซี่ฟันและหลาย lesion ซึ่งมี correlation สูงกัน หากใช้ random split แบบปกติ model อาจ "จำ" pattern เฉพาะของผู้ป่วยในชุด training ทำให้ test accuracy สูงเกินจริง (data leakage)

#### 5.1.2 class_weight='balanced'

`class_weight='balanced'` กำหนด weight ของแต่ละคลาสในสัดส่วนผกผันกับความถี่:

```
weight(c) = N / (n_classes × count(c))
```

เนื่องจาก caries บนผิว Occlusal พบน้อยกว่า Mesial/Distal ในธรรมชาติ การไม่ปรับ weight จะทำให้ model มี bias ต่อการทำนาย Mesial/Distal มากเกินไปและ Occlusal recall ต่ำมาก ซึ่งตรงกับปัญหาที่พบใน Baseline (Occlusal recall = 0.20) การใช้ balanced weight ช่วยยกระดับ Occlusal recall ไปถึง 0.84 ใน Run 3

#### 5.1.3 predict_proba แทน predict

ใน `classify_ml()` pipeline ไม่ใช้ `.predict()` โดยตรง แต่ใช้ `.predict_proba()` เพื่อดึงความน่าจะเป็นของแต่ละคลาสแยกกัน:

```python
class_probabilities = rf_model.predict_proba(prediction_input_df)[0]
surface_scores = {
    cls: class_probabilities[model_classes.index(cls)]
    for cls in ["Occlusal", "Mesial", "Distal"]
    if cls in model_classes
}
prediction = max(surface_scores, key=surface_scores.get)
```

วิธีนี้คือหัวใจของ **Smart Fallback** ซึ่งจะอธิบายในหัวข้อถัดไป

### 5.2 เหตุผลที่เลือกใช้ Random Forest

Random Forest ถูกเลือกสำหรับ dataset ขนาดนี้ด้วยเหตุผลหลายประการ:

1. **ทนทานต่อ overfitting**: การ ensemble trees หลายต้นและ bootstrap sampling ช่วยลด variance ทำให้ทำงานได้ดีบน dataset ขนาดกลาง (พัน samples)
2. **Multi-class native**: Random Forest รองรับ multi-class classification โดยตรงโดยไม่ต้อง decompose เป็น binary classifiers
3. **Feature importance**: ค่า Gini importance ช่วยตรวจสอบว่า feature ใดมีผลต่อการตัดสินใจ ซึ่งมีประโยชน์ต่อการ debug และการอ้างอิงทางการแพทย์
4. **รองรับ class_weight**: parameter นี้ถูก implement อย่างครบถ้วนสำหรับ multi-class ใน scikit-learn
5. **ไม่ต้อง normalization**: Random Forest ทำงานได้ดีกับ feature ที่มี scale ต่างกัน (เช่น `coverage` มีค่า 0–0.1 ส่วน `aspect_ratio` มีค่า 0.5–3) โดยไม่ต้องทำ StandardScaler ก่อน

### 5.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

**XGBoost** (Gradient Boosted Trees) มักให้ประสิทธิภาพสูงกว่า Random Forest บน tabular data เนื่องจากกระบวนการ boosting ที่ปรับ error แบบ sequential แต่ต้องการ hyperparameter tuning มากกว่า (`learning_rate`, `max_depth`, `min_child_weight`) และใช้เวลา train นานกว่า

**Support Vector Machine (SVM)** ด้วย RBF kernel เหมาะสำหรับ dataset ขนาดเล็กถึงกลาง แต่ multi-class SVM (One-vs-One) มี complexity O(N²) และช้ากว่า Random Forest มาก นอกจากนี้การ tune `C` และ `gamma` โดยไม่มี validation set ที่เหมาะสมทำได้ยาก

**Gradient Boosting (scikit-learn)** เป็นทางเลือกใกล้เคียง XGBoost แต่ช้ากว่า LightGBM/XGBoost หลายเท่า

**Multilayer Perceptron (MLP)** เป็นทางเลือกสำหรับ non-linear boundary ที่ซับซ้อน แต่สำหรับ 13 features และ training samples ระดับพันรายการ MLP มีความเสี่ยง overfit สูงและต้องการ regularization และ cross-validation อย่างระมัดระวัง

---

## 6. เทคนิคที่ 5 — Smart Fallback Architecture

### 6.1 หลักการทำงาน

Smart Fallback คือ design pattern สำคัญที่สุดของ Run 3 โดยทำงานผ่านลำดับชั้นดังนี้:

```
classify_ml()
    │
    ├─ [สำเร็จ] RF predict_proba → กรองเฉพาะ {Occlusal, Mesial, Distal}
    │           → เลือก class ที่มี probability สูงสุดใน 3 คลาสนั้น
    │
    └─ [ล้มเหลว / features = None / rf_model = None]
              → classify_xthird() (X-Thirds rule-based)
                      │
                      └─ [ล้มเหลว] → return "Other", 0.0, {}
```

จุดสำคัญคือ model ถูกฝึกด้วย 4 คลาสรวม "Other" แต่ระหว่าง inference **คลาส "Other" ถูกตัดออกจาก candidate set ตลอดเวลา** โดยใช้ `predict_proba` แล้วเลือก argmax เฉพาะใน `{Occlusal, Mesial, Distal}` เท่านั้น:

```python
surface_scores = {
    cls: class_probabilities[model_classes.index(cls)]
    for cls in ["Occlusal", "Mesial", "Distal"]  # ไม่มี "Other"
    if cls in model_classes
}
prediction = max(surface_scores, key=surface_scores.get)
```

### 6.2 เหตุผลที่เลือกใช้เทคนิคนี้

**ปัญหาของ "Other" class**: เนื่องจากเป้าหมายของงานคือการระบุผิวฟันที่มี caries จริง ๆ (Occlusal/Mesial/Distal) การที่ model ทำนายว่า "Other" จึงไม่มีประโยชน์ใด ๆ ต่อ clinical workflow ยิ่งไปกว่านั้น ในกรณีที่ model ไม่มั่นใจ (probability กระจายเท่า ๆ กันทั้ง 4 คลาส) การที่ "Other" ได้รับการเลือกแทนที่จะ force ให้เลือก 1 ใน 3 คลาสหลักถือเป็นผลลัพธ์ที่ไม่มีประโยชน์

**ผลที่ได้**: Smart Fallback เพิ่ม Mesial recall จาก ~0.77 ให้บรรลุเป้าหมาย 0.80+ และเพิ่ม Occlusal recall จาก 0.20 (Baseline) ไปสู่ 0.84

**Fallback สู่ X-Thirds**: X-Thirds classifier ถูกเก็บไว้เป็น safety net สำหรับกรณีที่ feature extraction ล้มเหลว (เช่น tooth mask มีขนาดเล็กเกินไป หรือ caries coordinates ว่างเปล่า) ทำให้ pipeline ไม่ crash และยังคืนค่า prediction ที่สมเหตุสมผลได้

### 6.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

**Hierarchical Classification**: สร้าง classifier 2 ขั้น — ขั้นแรก binary (caries vs no-caries) และขั้นสองตัดสิน surface ใน {Occlusal, Mesial, Distal} โดยตรง โดยไม่ใส่ "Other" ใน training set เลย วิธีนี้สะอาดกว่าเชิง conceptual แต่ต้องการ label "no-caries" ซึ่ง pipeline นี้ไม่มี เนื่องจาก input คือ tooth ที่มี caries แล้วเท่านั้น

**Cost-sensitive Learning**: ปรับ misclassification cost matrix แทน class_weight โดยกำหนดให้การทำนายผิดเป็น "Other" มี cost สูงกว่า วิธีนี้ต้องการ custom loss function ที่ Random Forest ใน scikit-learn ไม่ support โดยตรง

**Label Smoothing / Threshold Calibration**: ปรับ decision threshold สำหรับแต่ละคลาสแยกกัน (เช่น ถ้า P(Occlusal) > 0.25 ให้เลือก Occlusal แทนที่จะใช้ max) วิธีนี้ต้องการ calibration set และ การ tune threshold อย่างละเอียด

---

## 7. เทคนิคที่ 6 — X-Thirds Baseline Classifier

### 7.1 หลักการทำงาน

`classify_xthird()` เป็น rule-based classifier ที่แบ่งฟัน (หลัง PCA alignment) ออกเป็น 3 โซนตามแกน X:

```
Zone ซ้าย:  rel_x < 0.40  → Distal  (quadrant 1,4) / Mesial (quadrant 2,3)
Zone กลาง:  0.40 ≤ rel_x ≤ 0.60  → Occlusal
Zone ขวา:   rel_x > 0.60  → Mesial  (quadrant 1,4) / Distal  (quadrant 2,3)
```

จากนั้นนับว่า caries pixel อยู่ใน zone ไหนมากที่สุด (majority vote) แล้วคืนค่า winner เป็น surface prediction

Logic การสลับ Mesial/Distal ระหว่าง quadrant 1,4 กับ quadrant 2,3 สะท้อน FDI notation ที่ถูกต้อง โดย Mesial หมายถึงผิวที่หันเข้าหากึ่งกลางช่องปาก ซึ่งอยู่ฝั่งตรงข้ามกันระหว่างซ้ายและขวาของขากรรไกร

### 7.2 บทบาทในสถาปัตยกรรม

X-Thirds ไม่ได้เป็น production classifier หลัก แต่ทำหน้าที่สองประการ:

1. **Fallback สุดท้าย**: ถูกเรียกเมื่อ RF ล้มเหลวทุกกรณี ทำให้ระบบไม่เคยคืนค่าว่างเปล่า
2. **Benchmark อ้างอิง**: ผล Baseline (Occlusal recall 0.20) แสดงให้เห็นว่า rule-based approach ที่ไม่แยก lesion มีข้อจำกัดชัดเจน เนื่องจาก caries หลายจุดในฟันเดียวกัน เมื่อนำ centroid รวมมาใช้มักตกในโซนกลาง (Occlusal) ทั้งที่จริง ๆ เป็น Mesial หรือ Distal

---

## 8. เทคนิคที่ 7 — Streaming JSON Processing ด้วย ijson

### 8.1 หลักการทำงาน

Pipeline นี้ประมวลผลไฟล์ JSON ขนาดใหญ่ (ต่อ case สูงสุด 80MB+) โดยใช้ **incremental/streaming parser** ผ่าน `ijson.items()` แทนการโหลดไฟล์ทั้งหมดเข้า memory พร้อมกัน

```python
for tooth_dict in ijson.items(f, "teeth_data.item"):
    tid = str(tooth_dict.get("tooth_id", ""))
    coords = tooth_dict.get("pixel_coordinates", [])
    if tid and coords:
        seg_map[tid] = np.array(coords, dtype=np.float32)
    # tooth_dict ออกจาก scope → eligible for GC ทันที
```

Pattern นี้รับประกัน **O(1) peak memory per tooth** เนื่องจากมีเพียง 1 tooth dictionary อยู่ใน memory ณ เวลาใดเวลาหนึ่ง ส่วน tooth ก่อนหน้าได้รับการแปลงเป็น `np.float32 array` และเก็บใน `seg_map` แล้ว

นอกจากนี้ยังมี **File Size Guard** ที่ตรวจสอบขนาดไฟล์ก่อนการประมวลผลทุกครั้ง:

```python
if not _check_file_size(path, label):
    return  # skip files > 500MB (anomaly detection)
```

### 8.2 เหตุผลที่เลือกใช้เทคนิคนี้

สำหรับ 500 cases ที่แต่ละ case มีไฟล์ segmentation JSON ขนาด 40–80MB หากโหลดทั้งหมดพร้อมกันด้วย `json.load()` การใช้งาน RAM จะอยู่ที่หลายร้อย MB ต่อ case และการประมวลผลแบบ sequential จะทำให้ RAM ไม่ถูก free ระหว่าง iteration ส่งผลให้เกิด memory pressure สะสม Streaming parser แก้ปัญหานี้ได้ตรงจุด

การใช้ `gc.collect()` อย่างเป็น explicit หลังแต่ละ case เป็น defensive pattern ที่สำคัญเพื่อบังคับให้ Python GC เก็บ object ที่ไม่ได้ใช้แล้วอย่างแน่นอน โดยเฉพาะบน Windows ที่ GC อาจ delay มากกว่า Linux

### 8.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

หากสามารถ preprocess ข้อมูลได้ล่วงหน้า การแปลงไฟล์ JSON เป็น **Apache Arrow / Parquet format** จะให้ประสิทธิภาพการอ่านสูงกว่า ijson มากเนื่องจาก columnar storage รองรับ predicate pushdown และ lazy loading โดย library เช่น `polars` สามารถอ่าน Parquet แบบ lazy evaluation ได้โดยไม่โหลดทั้งไฟล์

อย่างไรก็ตามสำหรับ JSON ที่มี schema ไม่แน่นอน (เช่น field อาจขาดหายในบางกรณี) และไม่ต้องการ preprocess ขั้นตอนพิเศษ ijson เป็นตัวเลือกที่เหมาะสมที่สุด

---

## 9. เทคนิคที่ 8 — Greedy Multi-Lesion GT Matching

### 9.1 หลักการทำงาน

ฟังก์ชัน `match_case()` จับคู่ ground-truth surface กับ predicted surface ต่อ tooth โดยรองรับกรณีที่ tooth หนึ่งซี่มีหลาย lesion และหลาย GT surface พร้อมกัน:

```python
for gt_surf in gt_surfaces:
    if gt_surf in pred_surfaces:
        y_true.append(gt_surf); y_pred.append(gt_surf)
        pred_surfaces.remove(gt_surf)     # exact match ก่อน
    else:
        y_true.append(gt_surf)
        y_pred.append(pred_surfaces.pop(0) if pred_surfaces else "Other")
```

กลยุทธ์คือ **Greedy Exact Match First**: จับคู่ surface ที่ตรงกันก่อน จากนั้น GT ที่เหลือจับคู่กับ prediction ที่เหลือตามลำดับ และหากไม่มี prediction เหลือให้จับคู่กับ "Other"

### 9.2 เหตุผลที่เลือกใช้เทคนิคนี้

ในสถานการณ์จริง ฟัน 1 ซี่อาจมีทั้ง Occlusal และ Mesial caries พร้อมกัน GT จะมี 2 entries และ model จะ predict 2 lesions การ matching จึงต้องรองรับ many-to-many relationship ระหว่าง GT และ prediction Greedy matching เป็นวิธีที่ง่ายและ computationally inexpensive สำหรับปัญหานี้

### 9.3 ทางเลือกอื่นที่ดีกว่าหรือไม่

**Hungarian Algorithm** (optimal bipartite matching) จะให้การจับคู่ที่ optimal กว่า Greedy โดยมี guarantee ว่าผลรวมของ match score สูงสุด อย่างไรก็ตาม สำหรับ lesion count ต่อ tooth ที่มักไม่เกิน 3–4 ความแตกต่างระหว่าง Greedy และ Hungarian มีน้อยมาก และ Hungarian มี complexity O(N³) ซึ่งไม่คุ้มค่าสำหรับ N เล็กขนาดนี้

---

## 10. สรุปการประเมินเทคนิค

| เทคนิค | ความเหมาะสม | ทางเลือกที่น่าพิจารณา |
|---|---|---|
| PCA Alignment (4 กฎ) | เหมาะสม — สร้าง canonical space โดยไม่ต้องการ landmark annotation | Oriented Bounding Box (ง่ายกว่า แต่ยังต้องการ logic เพิ่ม) |
| DBSCAN (eps=2.0, min_samples=1) | เหมาะสม — แก้ปัญหา multi-lesion per tooth ได้ถูกต้อง | Connected Component Analysis (เร็วกว่ามากสำหรับ binary data) |
| 13 Geometric Features | เหมาะสม — interpretable, scale-invariant, เหมาะกับ dataset เล็ก | Zernike moments, CNN features (ต้องการ data มากกว่า) |
| Random Forest + class_weight='balanced' | เหมาะสม — robust, interpretable, native multi-class | XGBoost (มักดีกว่า แต่ต้องการ tuning) |
| GroupShuffleSplit by case_id | จำเป็นมาก — ป้องกัน data leakage ระดับผู้ป่วย | GroupKFold (cross-validation ครบถ้วนกว่า) |
| Smart Fallback (ตัด "Other") | เหมาะสม — แก้ปัญหา recall ต่ำของ Occlusal ได้ตรงจุด | Hierarchical 2-stage classifier (conceptually สะอาดกว่า) |
| ijson Streaming | เหมาะสม — ควบคุม memory สำหรับ JSON 500 files | Apache Arrow/Parquet (เร็วกว่าหากแปลง format ได้) |
| Greedy GT Matching | เพียงพอ — เหมาะกับ N เล็กต่อ tooth | Hungarian Algorithm (optimal แต่ไม่คุ้มค่าสำหรับ N ≤ 4) |

โดยรวม pipeline นี้แสดงให้เห็นถึงการออกแบบที่รอบคอบและมีเหตุผลอ้างอิงชัดเจนในทุก design decision ทั้งการเลือก algorithm การจัดการ memory และการจัดการ class imbalance เทคนิคที่ใช้ทั้งหมดเหมาะสมกับข้อจำกัดของ dataset ขนาด 500 รายที่มี annotation ในระดับ surface label เท่านั้น

---

*บันทึก: รายงานนี้วิเคราะห์จาก `pipeline_run3_final.py` (1,681 บรรทัด) ซึ่งเป็น Run 3 ของโครงการ ITDS346 Dental Caries Surface Classification*
