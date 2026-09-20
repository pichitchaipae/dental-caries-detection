import json

NB_PATH = "C:/Users/jaopi/Desktop/SP/phase2-1april/pipeline-phase1-v4.ipynb"
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][5]['source'])

# Fix corrupted format specifier
BROKEN = r":.# changed to tooth_pts2f"
FIXED  = r":.2f"

assert BROKEN in src, "Broken format string not found"
src = src.replace(BROKEN, FIXED)
print(f"[OK] Fixed {src.count(FIXED)} occurrence(s) of format specifier")

lines = src.split('\n')
nb['cells'][5]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
nb['cells'][5]['outputs'] = []
nb['cells'][5]['execution_count'] = None

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

with open(NB_PATH, encoding='utf-8') as f:
    nb2 = json.load(f)
src2 = ''.join(nb2['cells'][5]['source'])
print("[FAIL] Broken string still present" if BROKEN in src2 else "[OK] Format string fully fixed")
