"""Re-check the mesial/distal direction of the surface heuristic against XML.

For every labelled lesion in a sample of cases:

1. run Pano-Seg on the pano image;
2. match the XML tooth name -> FDI to the detected tooth;
3. compute ``x_rel`` in the tooth's PCA-aligned frame — the exact coordinate
   used by ``app/pipeline/surface_heuristic.py`` (and Stage 2's axes);
4. compare ``x_rel`` with the labelled surface.

The serving heuristic reads the refined Crop-Seg mask; this script uses the pano
mask because XML labels are in pano coordinates and no Crop-Seg weights are
needed for the direction check.  The direction is unchanged.

    python scripts/validate_surface_heuristic.py \
        --images-root "D:/Mahidol University/SP/data/500 cases with annotation" \
        --weights-dir weights --limit 120

Output: table on stdout + ``surface_validation.json`` next to this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.pca_alignment import perform_pca  # noqa: E402
from app.pipeline.xml_annotations import parse_case_dir  # noqa: E402


def _case_dirs(root: Path, limit: int) -> list[Path]:
    dirs = [d for d in root.glob("case *") if d.is_dir()]
    dirs.sort(key=lambda d: int(d.name.split()[1]))
    return dirs[:limit] if limit > 0 else dirs


def _case_image(case_dir: Path) -> Path | None:
    cid = case_dir.name.split()[1]
    candidate = case_dir / f"case_{cid}.png"
    if candidate.is_file():
        return candidate
    pngs = sorted(case_dir.glob("*.png"))
    return pngs[0] if pngs else None


def _rotate(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (points - center) @ rotation.T + center


def _aligned_x_rel(
    polygon: list[list[float]],
    fdi: int,
    point: tuple[float, float],
) -> float:
    tooth = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    center, angle, _ = perform_pca(tooth.tolist(), fdi)
    rotated = _rotate(tooth, center, angle)
    lesion = _rotate(np.asarray([point], dtype=np.float64), center, angle)[0]
    x_min = float(rotated[:, 0].min())
    x_max = float(rotated[:, 0].max())
    return (float(lesion[0]) - x_min) / max(1e-6, x_max - x_min)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-root", required=True)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--left-bound", type=float, default=0.30)
    p.add_argument("--right-bound", type=float, default=0.70)
    p.add_argument("--out", default="surface_validation.json")
    args = p.parse_args(argv)

    from ultralytics import YOLO

    weights = Path(args.weights_dir) / "Tooth_seg_pano_20250319.pt"
    if not weights.is_file():
        raise SystemExit(f"pano weights not found: {weights}")
    model = YOLO(str(weights))

    rows: list[dict] = []
    missing = defaultdict(int)

    for n, case_dir in enumerate(_case_dirs(Path(args.images_root), args.limit), 1):
        image_path = _case_image(case_dir)
        if image_path is None:
            continue
        lesions, _ = parse_case_dir(case_dir)
        if not lesions:
            continue

        results = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
        r = results[0]
        teeth: dict[int, dict] = {}
        if r.masks is not None:
            for i, box in enumerate(r.boxes):
                fdi = int(model.names[int(box.cls[0])])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                polygon = r.masks.xy[i].tolist() if i < len(r.masks.xy) else []
                teeth[fdi] = {"bbox": (x1, y1, x2, y2), "polygon": polygon}

        for lesion in lesions:
            tooth = teeth.get(lesion.fdi)
            if tooth is None:
                missing[lesion.fdi] += 1
                continue
            x1, y1, x2, y2 = tooth["bbox"]
            nx = (lesion.centroid[0] - x1) / max(1e-6, x2 - x1)
            ny = (lesion.centroid[1] - y1) / max(1e-6, y2 - y1)
            if len(tooth["polygon"]) < 3:
                continue
            try:
                rnx = _aligned_x_rel(tooth["polygon"], lesion.fdi, lesion.centroid)
            except Exception:
                continue
            rows.append({
                "case": int(case_dir.name.split()[1]),
                "fdi": lesion.fdi,
                "quadrant": lesion.fdi // 10,
                "surface": lesion.surface,
                "nx": round(float(nx), 4),
                "ny": round(float(ny), 4),
                "rnx": round(float(rnx), 4),
            })

        if n % 20 == 0:
            print(f"  {n} cases, {len(rows)} matched lesions", flush=True)

    if not rows:
        raise SystemExit("no matched lesions — check the image root and weights")

    print("\nPCA-aligned x (same coordinate as the serving heuristic):")
    print("quadrant surface      n    x_mean  x_median  left   mid    right")
    for q in (1, 2, 3, 4):
        for surface in ("mesial", "distal", "occlusal"):
            vals = [r["rnx"] for r in rows if r["quadrant"] == q and r["surface"] == surface]
            if not vals:
                continue
            left = sum(v < args.left_bound for v in vals) / len(vals)
            right = sum(v > args.right_bound for v in vals) / len(vals)
            mid = 1.0 - left - right
            print(
                f"Q{q}       {surface:8s} {len(vals):4d}   {np.mean(vals):7.3f}  "
                f"{np.median(vals):8.3f}  {left:.2f}  {mid:.2f}  {right:.2f}"
            )

    print("\ndirection check (proximal lesions only):")
    for q in (1, 4, 2, 3):
        mesial = [r["rnx"] for r in rows if r["quadrant"] == q and r["surface"] == "mesial"]
        distal = [r["rnx"] for r in rows if r["quadrant"] == q and r["surface"] == "distal"]
        if not mesial or not distal:
            continue
        if q in (1, 4):
            ok = (
                sum(v > args.right_bound for v in mesial) / len(mesial)
                + sum(v < args.left_bound for v in distal) / len(distal)
            ) / 2
            expected = "mesial right / distal left"
        else:
            ok = (
                sum(v < args.left_bound for v in mesial) / len(mesial)
                + sum(v > args.right_bound for v in distal) / len(distal)
            ) / 2
            expected = "mesial left / distal right"
        print(f"  Q{q}: {expected} — zone agreement {ok:.1%} (n={len(mesial)+len(distal)})")

    predictions = []
    for r in rows:
        q, x = r["quadrant"], r["rnx"]
        if x < args.left_bound:
            pred = "distal" if q in (1, 4) else "mesial"
        elif x > args.right_bound:
            pred = "mesial" if q in (1, 4) else "distal"
        else:
            pred = "occlusal"
        predictions.append(pred == r["surface"])
    print(f"\noverall 3-way agreement: {sum(predictions)}/{len(rows)} "
          f"({100.0 * sum(predictions) / len(rows):.1f}%)")

    payload = {
        "args": vars(args),
        "rows": rows,
        "unmatched_by_fdi": dict(missing),
        "overall_accuracy": sum(predictions) / len(rows),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
