"""
End-to-End Evaluation Pipeline (Full 5 Methods - Decayed Teeth in 500 Cases)
- ประเมินผลครบทั้ง 5 วิธี PCA (0, 1, 2, 3, 5)
- วิเคราะห์เฉพาะ "ซี่ฟันที่ผุ" ใน Case 1 - 500
- บันทึก Log แยกไฟล์ และจบโปรแกรมอัตโนมัติ
"""

import sys
import os
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =========================================================================
# 0. Path & Modules setup
# =========================================================================
# ชี้ไปที่โฟลเดอร์ week5 สำหรับ Classifier และ week6 สำหรับ Snodent Map
sys.path.insert(0, r"C:\Users\jaopi\Desktop\SP\week5") 
sys.path.insert(0, r"C:\Users\jaopi\Desktop\SP\week6") 

import caries_surface_classifier
from caries_surface_classifier import (
    classify_caries_surface_detailed,
    perform_pca_0, perform_pca_1, perform_pca_2, perform_pca_3, perform_pca_5
)
from snodent_tooth_map import SNODENT_TO_FDI, snodent_display_to_fdi

# =========================================================================
# 1. ฟังก์ชันโหลดข้อมูลจาก 3 แหล่ง
# =========================================================================
def load_all_tooth_polygons(json_folder):
    tooth_db = {}
    json_files = list(Path(json_folder).rglob("*_results.json"))
    for json_path in tqdm(json_files, desc="🦷 โหลดพิกัดฟัน (Week 2)", unit=" file"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                case_num = str(data.get('case_number', ''))
                if not case_num: continue
                if case_num not in tooth_db: tooth_db[case_num] = {}
                for tooth in data.get('teeth_data', []):
                    t_id = str(tooth.get('tooth_id', ''))
                    pixels = tooth.get('pixel_coordinates', [])
                    if t_id and pixels: tooth_db[case_num][t_id] = pixels
        except: pass
    return tooth_db

def load_all_caries_polygons(json_folder):
    caries_db = {}
    json_files = list(Path(json_folder).rglob("*_caries_mapping.json"))
    for json_path in tqdm(json_files, desc="🦠 โหลดพิกัดรอยผุ (Week 3)", unit=" file"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                case_num = str(data.get('case_number', ''))
                if not case_num: continue
                if case_num not in caries_db: caries_db[case_num] = {}
                for tooth in data.get('teeth_caries_data', []):
                    t_id = str(tooth.get('tooth_id', ''))
                    pixels = tooth.get('caries_coordinates', [])
                    if t_id and pixels: caries_db[case_num][t_id] = pixels
        except: pass
    return caries_db

def load_xml_ground_truth(xml_folder):
    gt_db = {}
    xml_files = list(Path(xml_folder).rglob("*.xml"))
    for xml_path in tqdm(xml_files, desc="🔗 อ่านเฉลย (XML)", unit=" file"):
        try:
            tree = ET.parse(xml_path); root = tree.getroot()
            ns = {'aim': 'gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM', 'iso': 'uri:iso.org:21090'}
            label = None; t_id = None
            for char in root.findall('.//aim:ImagingPhysicalEntityCharacteristic', ns):
                label_text = char.find('aim:label', ns).attrib.get('value', '').lower()
                type_code = char.find('aim:typeCode', ns)
                display_name = type_code.find('iso:displayName', ns).attrib.get('value', '').lower()
                if "what part of the region" in label_text:
                    if 'occlusal' in display_name: label = 0 
                    elif 'mesial' in display_name: label = 1
                    elif 'distal' in display_name: label = 2
                elif "what is the anatomical position" in label_text:
                    t_id = snodent_display_to_fdi(display_name)
            
            parent_folder = Path(xml_path).parent.name
            case_match = re.search(r'\d+', parent_folder)
            if case_match:
                case_num = case_match.group(0)
                if case_num not in gt_db: gt_db[case_num] = {}
                gt_db[case_num][str(t_id)] = label
        except: pass
    return gt_db

# =========================================================================
# 2. Main Process
# =========================================================================
if __name__ == "__main__":
    xml_path = r"C:\Users\jaopi\Desktop\SP\raw_data\500 cases with annotation(xml)"
    tooth_path = r"C:\Users\jaopi\Desktop\SP\week2-Tooth Detection & Segmentation\500-segmentation+recognition"
    caries_path = r"C:\Users\jaopi\Desktop\SP\week3-Caries-to-Tooth Mapping\dental_analysis_output"
    
    # กำหนดชื่อไฟล์ Log แบบไม่ซ้ำ
    counter = 1
    while os.path.exists(f"full_5methods_eval_500cases_{counter}.txt"): counter += 1
    log_file = f"full_5methods_eval_500cases_{counter}.txt"

    print(f"--- เริ่มประเมินผล 5 PCA Methods (เฉพาะฟันผุใน 500 เคสแรก) ---\n")
    
    tooth_db = load_all_tooth_polygons(tooth_path)
    caries_db = load_all_caries_polygons(caries_path)
    gt_db = load_xml_ground_truth(xml_path)
    
    # สร้างชุดทดสอบ: เฉพาะซี่ที่ผุใน Case 1 - 500
    test_cases = []
    all_cases_int = sorted([int(k) for k in tooth_db.keys()])
    target_cases = [str(c) for c in all_cases_int if c <= 500]
    
    for case_num in target_cases:
        if case_num not in gt_db: continue
        for t_id in gt_db[case_num]:
            if t_id in tooth_db.get(case_num, {}):
                test_cases.append({
                    'case_num': case_num,
                    't_id': t_id,
                    'tooth_poly': tooth_db[case_num][t_id],
                    'caries': caries_db.get(case_num, {}).get(t_id, []),
                    'expected': gt_db[case_num][t_id]
                })

    print(f"\n✅ พบฟันที่มีรอยผุทั้งหมด {len(test_cases)} ซี่ ใน 500 เคสแรก")

    target_names = ["Occlusal", "Mesial", "Distal"]
    label_ids = [0, 1, 2]

    # ครบทั้ง 5 Methods
    pca_methods = {
        'perform_pca_0': perform_pca_0,
        'perform_pca_1': perform_pca_1,
        'perform_pca_2': perform_pca_2,
        'perform_pca_3': perform_pca_3,
        'perform_pca_5': perform_pca_5
    }

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== Full 5-Method Evaluation (Case 1-500) ===\n")
        f.write(f"Total decayed teeth evaluated: {len(test_cases)}\n\n")

    for name, func in pca_methods.items():
        print(f"🚀 กำลังประมวลผล: {name}")
        caries_surface_classifier.perform_pca = func
        y_true, y_pred = [], []
        
        for t in tqdm(test_cases, desc=f"⏳ {name}", unit=" tooth"):
            if not t['caries']:
                pred = -1 # ทายผิดถ้าตรวจไม่เจอรอยผุ
            else:
                try:
                    res = classify_caries_surface_detailed(t['t_id'], t['tooth_poly'], t['caries'])
                    pred = res['classification']
                except: pred = -1
            
            y_true.append(t['expected'])
            y_pred.append(pred)

        report = classification_report(y_true, y_pred, target_names=target_names, labels=label_ids, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
        
        print(f"\nMethod: {name}")
        print(report)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"=== {name} ===\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"Report:\n{report}\n\n")

    print(f"\n🎉 การประเมินเสร็จสมบูรณ์ ผลลัพธ์อยู่ที่: {log_file}")
    sys.stdout.flush()
    os._exit(0)