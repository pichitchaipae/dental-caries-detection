import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def summarize_xml_annotations(xml_folder):
    # ตัวแปรเก็บสถิติ
    surface_counts = Counter()
    case_caries_count = Counter()
    total_annotations = 0
    
    # สแกนหาไฟล์ XML
    xml_files = list(Path(xml_folder).rglob("*.xml"))
    print(f"กำลังเริ่มสแกน XML ทั้งหมด {len(xml_files)} ไฟล์...\n")
    
    for xml_path in tqdm(xml_files, desc="📊 กำลังวิเคราะห์ Data", unit=" file"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {'aim': 'gme://caCORE.caCORE/4.4/edu.northwestern.radiology.AIM', 
                  'iso': 'uri:iso.org:21090'}
            
            # ดึงข้อมูล Case Number จากชื่อโฟลเดอร์
            parent_folder = Path(xml_path).parent.name
            case_match = re.search(r'\d+', parent_folder)
            case_id = case_match.group(0) if case_match else "Unknown"
            
            # ค้นหาลักษณะของพื้นผิว (Surface Characteristic)
            for char in root.findall('.//aim:ImagingPhysicalEntityCharacteristic', ns):
                label_elem = char.find('aim:label', ns)
                if label_elem is not None:
                    label_text = label_elem.attrib.get('value', '').lower()
                    
                    # ค้นหา displayName เพื่อระบุด้าน
                    type_code = char.find('aim:typeCode', ns)
                    if type_code is not None:
                        display_name_elem = type_code.find('iso:displayName', ns)
                        if display_name_elem is not None:
                            display_name = display_name_elem.attrib.get('value', '').lower()
                            
                            if "what part of the region" in label_text:
                                if 'occlusal' in display_name:
                                    surface_counts['Occlusal'] += 1
                                elif 'mesial' in display_name:
                                    surface_counts['Mesial'] += 1
                                elif 'distal' in display_name:
                                    surface_counts['Distal'] += 1
                                
                                total_annotations += 1
                                case_caries_count[case_id] += 1
        except Exception as e:
            continue

    # --- แสดงผลสรุป ---
    print("\n" + "="*40)
    print("📜 รายงานสรุปข้อมูล XML Ground Truth")
    print("="*40)
    print(f"จำนวนรอยผุรวมทั้งหมด: {total_annotations} จุด")
    print("-" * 40)
    
    for surface, count in surface_counts.items():
        percentage = (count / total_annotations) * 100
        print(f"📍 {surface:10}: {count:4} ซี่ ({percentage:.2f}%)")
    
    print("-" * 40)
    print(f"จำนวนเคส (Patient Cases) ทั้งหมดที่พบ: {len(case_caries_count)} เคส")
    avg_caries = total_annotations / len(case_caries_count) if case_caries_count else 0
    print(f"ค่าเฉลี่ยฟันผุต่อเคส: {avg_caries:.2f} ซี่")
    
    print("-" * 40)
    print("🏆 Top 5 เคสที่มีฟันผุมากที่สุด:")
    for case, count in case_caries_count.most_common(5):
        print(f"   Case {case}: {count} ซี่")
    print("="*40)

if __name__ == "__main__":
    xml_folder_path = r"C:\Users\jaopi\Desktop\SP\raw_data\500 cases with annotation(xml)"
    summarize_xml_annotations(xml_folder_path)