"""Build a 1-class YOLO-seg caries dataset from tooth crops.

Pipeline per case (same order as serving):

    pano image
      -> Pano-Seg YOLO (FDI + bbox + mask)
      -> dedupe (FDI, IoU > 0.5)
      -> crop with PAD 20
      -> Crop-Seg Detectron2 (optional but recommended)
      -> suppress other detected teeth inside the crop
      -> save the crop as the training image
      -> XML lesion polygons (pano) -> clip to crop -> normalise to 0..1

Only the XML polygons are used as **labels**.  No ROI coordinate is used as a
model input.

Example
-------
    python scripts/prepare_caries_crop_dataset.py \
        --images-root "D:/Mahidol University/SP/data/500 cases with annotation" \
        --weights-dir weights \
        --out datasets/caries_crop_v1 \
        --val-fraction 0.2 \
        --negatives-per-case 3 \
        --viz 24

Use ``--crop-seg off`` only when Detectron2 is unavailable; training on pano
masks then differs from serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow `python scripts/...` from the ml-service root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.crop_utils import (  # noqa: E402
    clip_polygon_to_crop,
    normalize_polygon,
    suppress_other_teeth,
)
from app.pipeline.detection import (  # noqa: E402
    assign_crop_bboxes,
    dedupe_detections,
    refine_with_crop_segmenter,
    run_pano_detection,
)
from app.pipeline.xml_annotations import parse_case_dir  # noqa: E402
from models.versions import MODEL_VERSIONS  # noqa: E402

log = logging.getLogger("prepare_caries_crop_dataset")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _case_dirs(root: Path, limit: int) -> list[Path]:
    dirs = [d for d in root.glob("case *") if d.is_dir()]

    def number(d: Path) -> int:
        try:
            return int(d.name.split()[1])
        except (IndexError, ValueError):
            return 10**9

    dirs.sort(key=number)
    if limit > 0:
        dirs = dirs[:limit]
    return dirs


def _case_image(case_dir: Path) -> Path | None:
    cid = case_dir.name.split()[1]
    candidate = case_dir / f"case_{cid}.png"
    if candidate.is_file():
        return candidate
    others = sorted(case_dir.glob("*.png"))
    return others[0] if others else None


def _write_dataset_yaml(out: Path) -> None:
    payload = {
        "path": out.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": ["caries"],
    }
    (out / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8",
    )


def _save_qc(
    path: Path,
    crop: np.ndarray,
    polygons: list[list[list[float]]],
    fdi: int,
) -> None:
    vis = crop.copy()
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        cv2.polylines(vis, [np.round(pts).astype(np.int32)], True, (0, 255, 0), 2)
    cv2.putText(
        vis, f"FDI {fdi}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    cv2.imwrite(str(path), vis)


def build(args: argparse.Namespace) -> int:
    images_root = Path(args.images_root)
    weights_dir = Path(args.weights_dir)
    out = Path(args.out)
    if not images_root.is_dir():
        raise SystemExit(f"images root not found: {images_root}")
    if not (weights_dir / MODEL_VERSIONS["pano_detector"]["filename"]).is_file():
        raise SystemExit(f"pano weights not found in {weights_dir}")

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty — pass --overwrite to rebuild")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if args.viz > 0:
        (out / "qc").mkdir(parents=True, exist_ok=True)

    # ---- models -----------------------------------------------------------
    from ultralytics import YOLO

    log.info("loading pano model: %s", MODEL_VERSIONS["pano_detector"]["filename"])
    pano = YOLO(str(weights_dir / MODEL_VERSIONS["pano_detector"]["filename"]))

    crop_bundle = None
    crop_mode = args.crop_seg
    if crop_mode in ("auto", "on"):
        try:
            from models.registry import _load_detectron2_crop
            crop_bundle = _load_detectron2_crop(weights_dir, device=args.device)
            crop_mode = "on"
        except Exception as exc:
            if args.crop_seg == "on":
                raise SystemExit(f"Crop-Seg required but failed to load: {exc}") from exc
            log.warning("Crop-Seg unavailable (%s) — falling back to pano masks", exc)
            crop_mode = "off"
    else:
        log.warning("Crop-Seg disabled — crops will use pano masks (train/serve gap)")

    # ---- cases ------------------------------------------------------------
    cases = _case_dirs(images_root, args.limit)
    rng = random.Random(args.seed)
    split_rng = random.Random(args.seed + 1)
    val_count = max(1, int(round(len(cases) * args.val_fraction))) if cases else 0
    val_cases = set(split_rng.sample([c.name for c in cases], val_count)) if val_count else set()

    stats = Counter()
    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "weights": {
            k: {"filename": v["filename"], "sha256": v["sha256"]}
            for k, v in MODEL_VERSIONS.items()
        },
        "crop_seg": crop_mode,
        "cases": {},
        "issues": [],
    }
    started = time.perf_counter()

    for n, case_dir in enumerate(cases, 1):
        cid = int(case_dir.name.split()[1])
        image_path = _case_image(case_dir)
        if image_path is None:
            manifest["issues"].append({"case": cid, "reason": "no png"})
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            manifest["issues"].append({"case": cid, "reason": "unreadable png"})
            continue

        lesions, xml_issues = parse_case_dir(case_dir)
        for issue in xml_issues:
            manifest["issues"].append({"case": cid, "reason": issue.reason, "file": issue.source_file})
        by_fdi: dict[int, list] = defaultdict(list)
        for lesion in lesions:
            by_fdi[lesion.fdi].append(lesion)

        detections = run_pano_detection(str(image_path), pano, args.detection_threshold)
        detections = dedupe_detections(detections, iou_thresh=args.dedupe_iou)
        assign_crop_bboxes(detections, image.shape[:2], pad=args.crop_pad)
        detections = refine_with_crop_segmenter(
            image, detections, crop_bundle, pad=args.crop_pad,
        )

        split = "val" if case_dir.name in val_cases else "train"
        negative_pool = [d for d in detections if not by_fdi.get(d.fdi)]
        negatives = set(
            d.fdi for d in rng.sample(
                negative_pool, min(len(negative_pool), args.negatives_per_case),
            )
        ) if args.negatives_per_case > 0 else set()

        case_stats = Counter()

        for det in detections:
            has_labels = bool(by_fdi.get(det.fdi))
            if not has_labels and det.fdi not in negatives:
                stats["crops_skipped_negative_sampling"] += 1
                continue
            if det.crop_bbox_xyxy is None:
                stats["crops_skipped_no_bbox"] += 1
                continue

            x1, y1, x2, y2 = det.crop_bbox_xyxy
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                stats["crops_skipped_empty"] += 1
                continue

            other_polygons = [
                other.tooth_polygon for other in detections if other is not det
            ]
            crop = suppress_other_teeth(
                crop, det.crop_bbox_xyxy, det.tooth_polygon, other_polygons,
            )

            label_lines: list[str] = []
            qc_polygons: list[list[list[float]]] = []
            for lesion in by_fdi.get(det.fdi, []):
                polygon_crop = clip_polygon_to_crop(
                    lesion.polygon, det.crop_bbox_xyxy, min_area=args.min_lesion_area,
                )
                if polygon_crop is None:
                    stats["lesions_skipped_outside_crop"] += 1
                    continue
                normalized = normalize_polygon(polygon_crop, crop.shape[:2])
                if len(normalized) < 3:
                    stats["lesions_skipped_degenerate"] += 1
                    continue
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
                label_lines.append(f"0 {coords}")
                qc_polygons.append(polygon_crop)
                stats["lesions_used"] += 1

            if has_labels and not label_lines:
                # The tooth is positive but no lesion is visible in this crop.
                # Do not save it as a negative — that would teach a false negative.
                stats["crops_skipped_positive_without_visible_lesion"] += 1
                continue

            stem = f"case{cid:03d}_fdi{det.fdi}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.png"), crop)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            stats[f"crops_{split}"] += 1
            stats["crops_positive" if label_lines else "crops_negative"] += 1
            case_stats["crops"] += 1
            case_stats["lesions"] += len(label_lines)

            if args.viz > 0 and stats["crops_positive"] <= args.viz and label_lines:
                _save_qc(
                    out / "qc" / f"{stem}.png",
                    crop, qc_polygons, det.fdi,
                )

        stats["cases_used"] += 1
        manifest["cases"][str(cid)] = {
            "split": split,
            "lesions_xml": len(lesions),
            "crops": case_stats["crops"],
            "lesions_used": case_stats["lesions"],
        }

        if n % 10 == 0 or n == len(cases):
            log.info(
                "%d/%d cases  crops=%d  positives=%d  lesions=%d  %.1fs",
                n, len(cases), stats["crops_train"] + stats["crops_val"],
                stats["crops_positive"], stats["lesions_used"],
                time.perf_counter() - started,
            )

    _write_dataset_yaml(out)
    manifest["stats"] = dict(stats)
    manifest["splits"] = {
        "train_cases": len(cases) - len(val_cases),
        "val_cases": len(val_cases),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )

    log.info(
        "done: %d crops (%d positive, %d negative), %d lesions in %.1fs",
        stats["crops_train"] + stats["crops_val"],
        stats["crops_positive"], stats["crops_negative"],
        stats["lesions_used"], time.perf_counter() - started,
    )
    log.info("dataset: %s", out.resolve())
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-root", required=True,
                   help="folder containing 'case N/case_N.png' and XML files")
    p.add_argument("--weights-dir", default="weights",
                   help="folder with Tooth_seg_pano_20250319.pt and Tooth_seg_crop_20250424.pth")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--val-fraction", type=float, default=0.20,
                   help="fraction of cases held out for validation (split by case)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="use only the first N cases (0 = all)")
    p.add_argument("--detection-threshold", type=float, default=0.25)
    p.add_argument("--crop-pad", type=int, default=20)
    p.add_argument("--dedupe-iou", type=float, default=0.50)
    p.add_argument("--crop-seg", choices=("auto", "on", "off"), default="auto",
                   help="Crop-Seg Detectron2: auto tries and falls back, on requires it")
    p.add_argument("--device", default="cpu")
    p.add_argument("--negatives-per-case", type=int, default=3,
                   help="sample this many lesion-free tooth crops per case (0 = none)")
    p.add_argument("--min-lesion-area", type=float, default=4.0,
                   help="minimum visible lesion area in crop pixels")
    p.add_argument("--viz", type=int, default=0,
                   help="write up to N QC images with labels drawn")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return build(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
