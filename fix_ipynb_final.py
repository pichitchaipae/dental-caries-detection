
import json

file_path = "c:/Users/jaopi/Desktop/SP/phase2-1april/pipeline-phase1-v3.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data["cells"]:
    if "source" in cell:
        src_str = "".join(cell["source"])
        if "def match_case(gt, pred):" in src_str:
            
            main_loop_code = """

# =========================================================
# MAIN LOOP
# =========================================================
all_y_true = []
all_y_pred = []

base_gt = Path("../data/500 cases with annotation")

for case_num in range(1, 501):

    gt_folder = base_gt / f"case {case_num}"

    gt = parse_case_ground_truth(gt_folder)
    pred = load_prediction(case_num)

    if len(gt) == 0 and len(pred) == 0:
        continue

    yt, yp = match_case(gt, pred)

    all_y_true.extend(yt)
    all_y_pred.extend(yp)


# =========================================================
# FINAL METRICS
# =========================================================
accuracy = accuracy_score(all_y_true, all_y_pred)

precision = precision_score(
    all_y_true,
    all_y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_y_true,
    all_y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_y_true,
    all_y_pred,
    average="macro",
    zero_division=0
)


# =========================================================
# PRINT FINAL EVALUATION
# =========================================================
print("\\n========== FINAL EVALUATION ==========")
print(f"Total Samples : {len(all_y_true)}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")


# =========================================================
# CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(
    all_y_true,
    all_y_pred,
    labels=VALID_SURFACES
)

cm_df = pd.DataFrame(
    cm,
    index=VALID_SURFACES,
    columns=VALID_SURFACES
)

print("\\n========== CONFUSION MATRIX ==========")
print(cm_df)

print("\\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        all_y_true,
        all_y_pred,
        labels=VALID_SURFACES,
        zero_division=0
    )
)
"""
            if "MAIN LOOP" not in src_str:
                new_src = src_str + main_loop_code
                cell["source"] = [line + "\\n" for line in new_src.split("\\n")]
                if cell["source"]:
                    cell["source"][-1] = cell["source"][-1][:-1]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

