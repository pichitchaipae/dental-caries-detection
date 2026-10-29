"""Train the 1-class crop-space caries YOLO-seg model.

The dataset is produced by ``prepare_caries_crop_dataset.py``:

    dataset/
      images/{train,val}/*.png     # tooth crops, neighbours suppressed
      labels/{train,val}/*.txt     # class 0 = caries, YOLO-seg polygons
      data.yaml

Example (CPU, smoke test):

    python scripts/train_caries_crop_seg.py \
        --data datasets/caries_crop_v1/data.yaml \
        --model yolov8n-seg.pt \
        --epochs 50 --imgsz 640 --batch 8 --device cpu

The script prints the exact ``versions.py`` entry (filename + SHA-256) for the
best checkpoint.  It does not edit ``models/versions.py`` automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_dataset(data_yaml: Path) -> dict:
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {data_yaml}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names")
    labels = list(names.values()) if isinstance(names, dict) else list(names or [])
    if labels != ["caries"]:
        raise SystemExit(
            f"{data_yaml} must be a 1-class 'caries' dataset; got names={labels!r}. "
            "The 3-class Occlusal/Proximal/Lingual taxonomy was rejected."
        )
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    for split in ("train", "val"):
        rel = payload.get(split)
        if rel is None:
            raise SystemExit(f"{data_yaml} is missing '{split}'")
        split_dir = root / rel
        if not split_dir.is_dir():
            raise SystemExit(f"{split} images directory not found: {split_dir}")
    return payload


def train(args: argparse.Namespace) -> int:
    _check_dataset(Path(args.data))

    from ultralytics import YOLO

    model_path = args.model
    if not Path(model_path).is_file():
        # Ultralytics downloads the pretrained checkpoint by name.
        print(f"[train] pretrained backbone will be fetched: {model_path}")

    model = YOLO(model_path)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        task="segment",
        exist_ok=args.exist_ok,
        plots=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training finished but best.pt not found: {best}")

    metrics = model.val(data=args.data, split="val", device=args.device, verbose=False)
    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)
    print("\n=== validation ===")
    if box is not None:
        print(f"box  mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}")
    if seg is not None:
        print(f"seg  mAP50={seg.map50:.4f}  mAP50-95={seg.map:.4f}")

    target = Path(args.export_to) if args.export_to else best
    if args.export_to:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)

    digest = _sha256(target)
    suggested = f"caries_crop_seg_{date.today():%Y%m%d}.pt"
    print("\n=== checkpoint ===")
    print(f"best.pt : {best}")
    print(f"exported: {target}")
    print(f"sha256  : {digest}")
    print("\nAdd this to models/versions.py after verifying the file:\n")
    print(f'''    "caries_detector": {{
        "version": "caries-crop-seg-{date.today():%Y%m%d}",
        "filename": "{target.name if args.export_to else suggested}",
        "sha256": "{digest}",
        "architecture": "yolo_segment",
        "num_classes": 1,
        "classes": ["caries"],
        "description": "1-class caries YOLO-seg trained on tooth crops (PAD {args.crop_pad})",
    }},''')
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--model", default="yolov8n-seg.pt",
                   help="pretrained YOLO-seg checkpoint (n/s/m) or a local .pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu", help="cpu, 0, 0,1, ...")
    p.add_argument("--project", default="runs/caries_crop_seg")
    p.add_argument("--name", default="train")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--crop-pad", type=int, default=20,
                   help="only used in the printed versions.py description")
    p.add_argument("--export-to", default="",
                   help="optional path to copy best.pt to (e.g. weights/caries_crop_seg_20261028.pt)")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return train(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
