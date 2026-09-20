
import json

file_path = "c:/Users/jaopi/Desktop/SP/phase2-1april/pipeline-phase1-v3.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data["cells"]:
    if "source" in cell:
        src_str = "".join(cell["source"])
        if "def match_case(gt, pred):" in src_str:
            head = src_str.split("def match_case(gt, pred):")[0]
            tail = src_str.split("def match_case(gt, pred):")[1]
            
            # Find the end of the function (a def or # =========)
            import re
            end_match = re.search(r"\n\n\n# ====|\n# ====", tail)
            if end_match:
                end_idx = end_match.start()
                tail_after_func = tail[end_idx:]
            else:
                tail_after_func = ""
                
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

    return y_true, y_pred"""

            new_src = head + "def match_case(gt, pred):" + new_func_body + tail_after_func
            cell["source"] = [line + "\n" for line in new_src.split("\n")]
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1][:-1]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

