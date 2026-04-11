import json
file_path = "c:/Users/jaopi/Desktop/SP/phase2-1april/pipeline-phase1-v3.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data["cells"]:
    if "source" in cell and any("def match_case" in line for line in cell["source"]):
        new_source = []
        skip = False
        for line in cell["source"]:
            if "# 2. Catch False Positives" in line:
                skip = True
            if skip and "return y_true, y_pred" in line:
                skip = False
                new_source.append("    return y_true, y_pred\n")
                continue
            if not skip:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

