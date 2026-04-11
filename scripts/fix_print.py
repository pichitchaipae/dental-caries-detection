
import json

file_path = "c:/Users/jaopi/Desktop/SP/phase2-1april/pipeline-phase1-v3.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data["cells"]:
    if "source" in cell:
        for i, line in enumerate(cell["source"]):
            if "print(\"\\" in line and "========== " not in line:
                pass
            if line.startswith("print(\"\\n\\n"):
                 pass # skip
                 
            # Let"s just do a simple string replace
        src = "".join(cell["source"])
        src = src.replace("print(\"\\n\n========== FINAL EVALUATION ==========\")", "print(\"\\n========== FINAL EVALUATION ==========\")")
        src = src.replace("print(\"\\n\n========== CONFUSION MATRIX ==========\")", "print(\"\\n========== CONFUSION MATRIX ==========\")")
        src = src.replace("print(\"\\n\n========== CLASSIFICATION REPORT ==========\")", "print(\"\\n========== CLASSIFICATION REPORT ==========\")")
        
        # split back
        lines = [l + "\n" for l in src.split("\n")]
        if lines:
            lines[-1] = lines[-1][:-1]
        cell["source"] = lines

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

