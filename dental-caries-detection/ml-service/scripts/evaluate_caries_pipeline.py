"""Held-out evaluation + confidence calibration for the crop-space caries model.

Runs the real serving pipeline (Pano-Seg -> Crop-Seg -> 1-class caries YOLO on
the tooth crop) on the validation cases recorded in the dataset manifest, then
matches predicted lesion masks to the XML lesion polygons.

One inference pass is done at a low ``--collect-conf``; precision/recall/F1 are
then swept offline over ``--thresholds`` so calibration does not require
re-running the pipeline per threshold.

Example (inside the ml-service image):

    python scripts/evaluate_caries_pipeline.py \
        --dataset /out/caries_crop_v1 \
        --images-root "/data/500 cases with annotation" \
        --weights-dir /work/weights \
        --caries-weights /work/weights/caries_crop_seg_20261028.pt \
        --device cuda \
        --out /runs/caries_eval.json

The matching rule is greedy per FDI: a prediction is a TP when its mask IoU
with an unmatched GT lesion on the same tooth is >= ``--match-iou``.  Surface
agreement is computed only for matched lesions, using the same deterministic
heuristic as serving.  XML labels are never used as model input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.detection import run_stage1  # noqa: E402
from app.pipeline.surface_heuristic import classify_lesion_surface  # noqa: E402
from app.pipeline.xml_annotations import parse_case_dir  # noqa: E402


def _case_dirs(root: Path, val_ids: set[int] | None, limit: int) -> list[Path]:
    dirs = [d for d in root.glob("case *") if d.is_dir()]
    dirs.sort(key=lambda d: int(d.name.split()[1]))
    if val_ids is not None:
        dirs = [d for d in dirs if int(d.name.split()[1]) in val_ids]
    return dirs[:limit] if limit > 0 else dirs


def _case_image(case_dir: Path) -> Path | None:
    cid = case_dir.name.split()[1]
    candidate = case_dir / f"case_{cid}.png"
    if candidate.is_file():
        return candidate
    others = sorted(case_dir.glob("*.png"))
    return others[0] if others else None


def _bbox(polygon: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(polygon[:, 0].min()),
        float(polygon[:, 1].min()),
        float(polygon[:, 0].max()),
        float(polygon[:, 1].max()),
    )


def _bbox_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_iou(poly_a, poly_b, bbox_a=None, bbox_b=None) -> float:
    """IoU of two pano-space polygons, rasterized in their local union bbox."""
    a = np.asarray(poly_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(poly_b, dtype=np.float64).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ba = bbox_a or _bbox(a)
    bb = bbox_b or _bbox(b)
    if _bbox_iou(ba, bb) <= 0.0:
        return 0.0

    x1 = int(np.floor(min(ba[0], bb[0]))) - 1
    y1 = int(np.floor(min(ba[1], bb[1]))) - 1
    x2 = int(np.ceil(max(ba[2], bb[2]))) + 1
    y2 = int(np.ceil(max(ba[3], bb[3]))) + 1
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return 0.0

    def raster(poly: np.ndarray) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        pts = np.round(poly - [x1, y1]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
        return m

    ma, mb = raster(a), raster(b)
    union = int(np.logical_or(ma, mb).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(ma, mb).sum()) / union


def _match(
    gt_by_fdi: dict[int, list],
    pred_by_fdi: dict[int, list],
    match_iou: float,
) -> list[tuple]:
    """Greedy per-FDI matching.  Returns matched ``(gt, pred, iou)`` pairs."""
    pairs: list[tuple] = []
    for fdi in sorted(set(gt_by_fdi) | set(pred_by_fdi)):
        gts = gt_by_fdi.get(fdi, [])
        preds = pred_by_fdi.get(fdi, [])
        used: set[int] = set()
        for gt in gts:
            best_i, best_iou = -1, match_iou
            for i, pred in enumerate(preds):
                if i in used:
                    continue
                if _bbox_iou(gt["bbox"], pred["bbox"]) <= 0.0:
                    continue
                iou = mask_iou(gt["polygon"], pred["polygon"], gt["bbox"], pred["bbox"])
                if iou >= best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0:
                used.add(best_i)
                pairs.append((gt, preds[best_i], best_iou))
    return pairs


def _counts(gt_by_fdi: dict[int, list], pred_by_fdi: dict[int, list], pairs: list) -> tuple[int, int, int]:
    n_gt = sum(len(v) for v in gt_by_fdi.values())
    n_pred = sum(len(v) for v in pred_by_fdi.values())
    tp = len(pairs)
    return tp, n_pred - tp, n_gt - tp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset root containing manifest.json")
    p.add_argument("--images-root", required=True)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--caries-weights", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int, default=0, help="max val cases (0 = all)")
    p.add_argument("--detection-threshold", type=float, default=0.25)
    p.add_argument("--collect-conf", type=float, default=0.01,
                   help="low confidence for the single collection pass")
    p.add_argument("--match-iou", type=float, default=0.30)
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60")
    p.add_argument("--out", default="caries_eval.json")
    args = p.parse_args(argv)

    from ultralytics import YOLO
    from models.registry import _load_detectron2_crop

    dataset = Path(args.dataset)
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_ids = {
        int(cid) for cid, info in manifest.get("cases", {}).items()
        if info.get("split") == "val"
    }
    if not val_ids:
        raise SystemExit("no val cases in manifest.json")

    weights = Path(args.weights_dir)
    models = {
        "pano": YOLO(str(weights / "Tooth_seg_pano_20250319.pt")),
        "crop": _load_detectron2_crop(weights, device=args.device),
        "caries": YOLO(args.caries_weights),
    }

    cases = _case_dirs(Path(args.images_root), val_ids, args.limit)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    per_case: list[dict] = []
    for n, case_dir in enumerate(cases, 1):
        image_path = _case_image(case_dir)
        if image_path is None:
            continue
        cid = int(case_dir.name.split()[1])
        gt_lesions, issues = parse_case_dir(case_dir)
        dets, _ = run_stage1(
            str(image_path),
            models,
            detection_threshold=args.detection_threshold,
            caries_conf=args.collect_conf,
            suppress_neighbours=True,
        )

        gt_by_fdi: dict[int, list] = defaultdict(list)
        for lesion in gt_lesions:
            poly = np.asarray(lesion.polygon, dtype=np.float64).reshape(-1, 2)
            if len(poly) < 3:
                continue
            gt_by_fdi[lesion.fdi].append({
                "fdi": lesion.fdi,
                "surface": lesion.surface,
                "polygon": poly,
                "bbox": _bbox(poly),
            })

        pred_by_fdi: dict[int, list] = defaultdict(list)
        for det in dets:
            for lesion in det.caries_lesions:
                poly = np.asarray(lesion.polygon, dtype=np.float64).reshape(-1, 2)
                if len(poly) < 3:
                    continue
                pred_by_fdi[det.fdi].append({
                    "fdi": det.fdi,
                    "confidence": float(lesion.confidence),
                    "centroid": lesion.centroid,
                    "tooth_polygon": det.tooth_polygon,
                    "polygon": poly,
                    "bbox": _bbox(poly),
                })

        per_case.append({
            "case": cid,
            "gt_by_fdi": gt_by_fdi,
            "pred_by_fdi": pred_by_fdi,
            "xml_issues": len(issues),
            "teeth": len(dets),
        })
        if n % 10 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)} val cases", flush=True)

    sweep = []
    for threshold in thresholds:
        tp = fp = fn = 0
        surface_ok = surface_n = 0
        confusion: Counter = Counter()
        for case in per_case:
            preds = {
                fdi: [p for p in plist if p["confidence"] >= threshold]
                for fdi, plist in case["pred_by_fdi"].items()
            }
            preds = {fdi: plist for fdi, plist in preds.items() if plist}
            pairs = _match(case["gt_by_fdi"], preds, args.match_iou)
            c_tp, c_fp, c_fn = _counts(case["gt_by_fdi"], preds, pairs)
            tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
            for gt, pred, _iou in pairs:
                name = classify_lesion_surface(
                    gt["fdi"], pred["tooth_polygon"], pred["centroid"],
                )["name"]
                confusion[(gt["surface"], name)] += 1
                surface_n += 1
                surface_ok += int(name == gt["surface"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "surface_accuracy": round(surface_ok / surface_n, 4) if surface_n else None,
            "surface_n": surface_n,
        })

    best = max(sweep, key=lambda r: (r["f1"], r["recall"]))

    print("\nthreshold   TP    FP    FN   precision  recall     F1   surface_acc (n)")
    for row in sweep:
        acc = "—" if row["surface_accuracy"] is None else f"{row['surface_accuracy']:.3f}"
        print(
            f"  {row['threshold']:.2f}   {row['tp']:4d}  {row['fp']:4d}  {row['fn']:4d}"
            f"    {row['precision']:.3f}    {row['recall']:.3f}   {row['f1']:.3f}"
            f"    {acc} ({row['surface_n']})"
        )
    print(f"\nbest F1: threshold={best['threshold']:.2f}  F1={best['f1']:.3f}  "
          f"P={best['precision']:.3f}  R={best['recall']:.3f}")

    # Surface confusion at the selected threshold.
    threshold = best["threshold"]
    confusion: Counter = Counter()
    for case in per_case:
        preds = {
            fdi: [p for p in plist if p["confidence"] >= threshold]
            for fdi, plist in case["pred_by_fdi"].items()
        }
        preds = {fdi: plist for fdi, plist in preds.items() if plist}
        for gt, pred, _iou in _match(case["gt_by_fdi"], preds, args.match_iou):
            name = classify_lesion_surface(
                gt["fdi"], pred["tooth_polygon"], pred["centroid"],
            )["name"]
            confusion[(gt["surface"], name)] += 1
    print("\nsurface confusion (XML -> heuristic):")
    for gt_s in ("mesial", "occlusal", "distal"):
        row = {pred_s: confusion[(gt_s, pred_s)] for pred_s in ("mesial", "occlusal", "distal")}
        print(f"  {gt_s:8s} -> {row}")

    payload = {
        "args": vars(args),
        "val_cases": len(per_case),
        "match_iou": args.match_iou,
        "thresholds": sweep,
        "selected": best,
        "surface_confusion": {
            f"{gt}->{pred}": count for (gt, pred), count in sorted(confusion.items())
        },
        "per_case": [
            {
                "case": c["case"],
                "teeth": c["teeth"],
                "gt_lesions": sum(len(v) for v in c["gt_by_fdi"].values()),
                "pred_lesions_at_selected": sum(
                    len([p for p in v if p["confidence"] >= threshold])
                    for v in c["pred_by_fdi"].values()
                ),
            }
            for c in per_case
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
