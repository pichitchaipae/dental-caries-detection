# Progress Backend — ภาคเรียน 2 Sprint 1

> วันที่: 2026-11-14 · ผู้รับผิดชอบ: ไผ่ (Backend)
> Branch: `feat/sprint_1_sem2` · สถานะ: ทำใน working tree แล้ว ยังไม่ stage/commit/push
> อ้างอิง: [API contract v2 draft](api-contract-v2-draft.md) §2.3

## งานที่เสร็จแล้ว

ข้อ 3–9 ของรายการงาน BE ทำครบในขอบเขต Backend แล้ว แต่ contract ยังเป็น draft และยังไม่ได้ทดสอบร่วมกับ FE/ML แบบ end-to-end

| ข้อ | งาน | ผลที่ทำ |
|---|---|---|
| 3 | รับ `caries[].surface` และ `id` | รับจาก ML; ไม่มี surface ส่ง `null`; ไม่มี ID สร้าง integer ที่ไม่ซ้ำภายในซี่; ID ซ้ำจาก ML ถูกปฏิเสธ |
| 4 | เปลี่ยนชื่อ confidence | แปลง raw `caries[].confidence` เป็น public `caries[].probability` |
| 5 | ตัด field ที่ไม่ใช้ | ไม่ส่ง `teeth[].confidence` และ `caries[].bbox` ให้ FE; ยังเก็บ tooth bbox ไว้ตามข้อเสนอใน draft |
| 6 | default caries | ส่ง `caries: []` เมื่อ raw ไม่มีรายการ |
| 7 | ตรวจ surface | ถ้าเป็นชื่อด้านที่รองรับแต่ไม่อยู่ใน `surfaces[]` ของซี่นั้น ให้ request logger บันทึก warning และยังส่งผล; surface ที่หายไป/null ส่ง null; ชื่อด้านนอก vocabulary ยังไม่ผ่าน validation |
| 8 | fixture จากผล ML จริง | เปลี่ยน raw/public fixture เป็นผล case 10 ที่บันทึกไว้ พร้อมบันทึกที่มาและการปรับเป็น v2 |
| 9 | unit test | ครบไม่มีรอยผุ, หลายรอยผุด้านเดียวกัน, ไม่มี surface และ confidence แบบเดิม รวมถึง ID, warning, fixture parity และพิกัด mask |

แก้ route `GET /process` ให้ส่ง request logger เข้า adapter และเพิ่ม block ของ `case 'done'` เพื่อให้ผ่าน lint ด้วย

## ข้อมูล fixture

- ใช้ saved inference วันที่ 2026-10-29 จาก `case10_native_service.json` และ `case10_native_research.json` ใน `outputs/notebook-final-review-20261029/`
- มีฟัน 29 ซี่ และรอยผุที่โมเดลทำนาย 2 จุด: FDI 45 และ 26 ด้าน distal มี polygon 20 และ 59 จุดตามลำดับ
- คงพิกัด mask, bbox, centroid และ confidence จากผล service โดยไม่ปัดเศษเพิ่ม; ตรวจความตรงกันกับผล research ก่อนใช้ surface ที่บันทึกไว้
- เพิ่ม lesion ID เป็นลำดับภายในซี่เพื่อจำลอง ML-10 และปรับ surface probability จากค่า null ในผลเก่าเป็นคะแนนสูงสุดของ lesion ในด้านนั้นตาม draft
- สร้าง public fixture ผ่าน adapter จาก raw fixture ไม่ได้รัน inference ใหม่
- `image_base64` ยังเป็นภาพ placeholder 1×1 ไม่ใช่ภาพ case 10 จึงยังใช้ตรวจ overlay กับภาพจริงไม่ได้; พิกัด mask อ้างอิงภาพเดิม 3037×1536

รายละเอียดไฟล์ต้นทางและ SHA-256 อยู่ใน [fixture README](../backend/src/fixtures/README.md)

## ผลตรวจล่าสุด

| รายการ | ผล |
|---|---|
| `npm test` | ผ่าน 29 tests; ข้าม 10 tests ของ PostgreSQL integration เพราะไม่ได้ตั้ง `TEST_DATABASE_URL` |
| `npm run typecheck` | ผ่าน |
| `npm run lint` | ผ่าน |
| `npm run build` | ผ่านหลังปรับ adapter/route; การอัปเดต fixture รอบล่าสุดตรวจด้วย tests และ typecheck |
| Prettier เฉพาะไฟล์ที่แก้ | ผ่าน |
| `git diff --check` | ผ่าน |

Format check ทั้งโปรเจกต์ในรอบก่อนพบ 9 ไฟล์เดิมที่ไม่ผ่าน ซึ่งไม่ได้แก้ในงานนี้ ผลผ่านด้าน format จึงหมายถึงไฟล์ที่แก้เท่านั้น ยังไม่ได้ตรวจ end-to-end ด้วย model inference ใหม่หรือฐานข้อมูลจริง

## งานที่ยังรอ

1. ตกลงกับปอนและเจาเรื่องเก็บ `teeth[].bbox`, ส่ง mask ฟันที่ไม่ผุ และชื่อ Probability บนเว็บ รวมคำอธิบายว่าเป็นคะแนนโมเดลที่ยังไม่ calibrated
2. ขอเจาทำ ML-10 ให้ runtime ส่ง `surface` และ `id` ราย lesion จริง; ตอนนี้ BE รองรับ fallback แล้ว แต่ยังไม่ได้ติดต่อเจาหรือแก้ ML
3. ให้ปอนปรับ FE schema/view model และการวาด mask ให้รองรับ v2; FE ปัจจุบันยังคาดหวัง field แบบ v1
4. คัดลอก fixture ให้ FE แบบ byte-identical ตามกฎ contract — ยังไม่ได้ทำ
5. Freeze เป็น v2.0 และเปิด PR ประสานกับ FE เมื่อทีมตกลงแล้ว — ยังไม่ได้ทำ

งานรองรับหลายผู้ใช้, ลด polling, เปลี่ยนการเก็บสถานะใน SQL และออกแบบ job API ใหม่ยังอยู่นอก Sprint นี้ ตาม §4 ของ draft
