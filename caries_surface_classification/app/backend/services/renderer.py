"""
Renderer — Annotated Panoramic Radiograph
==========================================
Draws bounding boxes on the panoramic X-ray:
  • Red   (#FF4444) + surface label  → teeth WITH caries findings
  • Gray  (#AAAAAA)                  → all other detected teeth (no caries)

Arguments:
    image_path    : path to the source panoramic image.
    findings      : list of FinalFinding dicts  (caries teeth — from RF pipeline).
    all_tooth_bboxes : list of [x1,y1,x2,y2,fdi] for ALL detected teeth
                       (used to draw the grey "no-caries" boxes).
    output_dir    : directory to write annotated.png into.
"""

import cv2
import os


def render_predictions(
    image_path: str,
    findings: list,
    output_dir: str,
    all_tooth_bboxes: list = None,
):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # ── Build a set of caries FDI+bbox for fast lookup ──────────────
    caries_finding_ids = set()
    for f in findings:
        caries_finding_ids.add(tuple(f["bbox"]))

    font          = cv2.FONT_HERSHEY_SIMPLEX
    font_scale    = 0.55
    font_thickness = 2
    line_height   = 20

    # ── Draw grey boxes for non-caries teeth first (so red is on top) ─
    if all_tooth_bboxes:
        for tooth in all_tooth_bboxes:
            x1, y1, x2, y2, fdi = tooth
            # Skip if this tooth is a caries finding (will be drawn in red below)
            if tuple([x1, y1, x2, y2]) in caries_finding_ids:
                continue
            color = (160, 160, 160)  # grey
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label = f"[{fdi}]"
            (tw, th), _ = cv2.getTextSize(label, font, 0.42, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(img, label, (x1 + 1, y1 - 2), font, 0.42, (50, 50, 50), 1, cv2.LINE_AA)

    # ── Draw red boxes for caries findings ──────────────────────────
    for pred in findings:
        x1, y1, x2, y2 = pred["bbox"]
        color     = (60, 60, 220)   # BGR red
        thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label_lines = []
        if "tooth" in pred:
            label_lines.append(f"[{pred['tooth']}]")
        surface = pred["surface"].replace("_", " ").title()
        label_lines.append(surface)
        conf_pct = int(pred.get("confidence", 0) * 100)
        label_lines.append(f"{conf_pct}%")

        text_y = max(y1 - (len(label_lines) * line_height), 0)
        for idx, line in enumerate(label_lines):
            y_offset = text_y + (idx * line_height)
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            cv2.rectangle(img, (x1, y_offset - text_h - 4), (x1 + text_w, y_offset + 4), color, -1)
            cv2.putText(img, line, (x1, y_offset), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    output_filename = "annotated.png"
    output_path     = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img)
    return f"outputs/{output_filename}"
