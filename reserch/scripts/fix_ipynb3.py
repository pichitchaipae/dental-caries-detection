
import json

file_path = "c:/Users/jaopi/Desktop/SP/phase2-1april/pipeline-phase1-v3.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data["cells"]:
    if "source" in cell and any("def match_case" in line for line in cell["source"]):
        src = "".join(cell["source"])
        
        # We will split at def match_case(gt, pred):
        head, tail = src.split("def match_case(gt, pred):")
        
        # split tail at MAIN LOOP
        tail1, tail2 = tail.split("# MAIN LOOP")
        
        new_func_body = """
    # Create a mapping for prediction (1 prediction per tooth usually)
    pred_dict = {
        p["tooth"]: p["surface"]
        for p in pred
    }

    y_true = []
    y_pred = []

    # 1. Iterate through every single Ground Truth lesion (Lesion-level)
    for g in gt:
        tooth = g["tooth"]
        gt_surface = g["surface"]
        
        # If the model has a prediction for this tooth, use it; else "Other"
        pred_surface = pred_dict.get(tooth, "Other")
        
        y_true.append(gt_surface)
        y_pred.append(pred_surface)

    return y_true, y_pred


# =========================================================
"""
        new_src = head + "def match_case(gt, pred):" + new_func_body + "# MAIN LOOP" + tail2
        
        lines = [line + "\n" for line in new_src.split("\n")]
        if lines:
            lines[-1] = lines[-1][:-1]
            
        cell["source"] = lines

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

