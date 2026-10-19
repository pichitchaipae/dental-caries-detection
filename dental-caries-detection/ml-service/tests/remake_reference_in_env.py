"""Rebuild the parity reference INSIDE the current environment.

Why: the shipped ``parity_reference.json`` is produced on the HOST, so comparing
it inside a container mixes two different things
  (a) vendored code vs original code, and
  (b) the host OpenCV/torch build vs the container build.
The oracle check in ``test_surface_cnn_parity.py`` already isolates (a); this
script isolates (b) by regenerating the reference with the ORIGINAL
``pipeline.py`` running in the SAME environment, then the parity test can be
re-run against it with an exact-hash gate.

It re-uses the crop/mask PNG bytes already embedded in an existing reference
file, so no extra patient data is copied anywhere and no cache access is needed.

Run inside the ml-service container (CARIES_PIPELINE_DIR = folder of pipeline.py):

    python /tmp/psd/remake_reference_in_env.py \
        --src /weights/cnn_pca42/parity_reference.json \
        --checkpoints /weights/cnn_pca42 \
        --out /tmp/psd/parity_reference.inenv.json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_SURFACES = ["Distal", "Mesial", "Occlusal"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="existing parity_reference.json")
    ap.add_argument("--checkpoints", required=True, help="folder with best_pca_s42_f*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline-dir", default=os.environ.get("CARIES_PIPELINE_DIR", ""))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    root = a.pipeline_dir.strip()
    if not root or not (Path(root) / "pipeline.py").is_file():
        print(f"[in-env ref] FAIL pipeline.py not found in {root!r}")
        return 1
    if root not in sys.path:
        sys.path.insert(0, root)

    import cv2
    import torch

    import pipeline as P  # original training code, loaded from THIS environment

    src = json.loads(Path(a.src).read_text(encoding="utf-8"))
    cfg = src["config"]

    cps = sorted(Path(a.checkpoints).glob("best_pca_s42_f[0-9].pt"))
    if not cps:
        print(f"[in-env ref] FAIL no checkpoints in {a.checkpoints}")
        return 1

    models = []
    for cp in cps:
        ck = torch.load(str(cp), map_location="cpu", weights_only=False)
        net = P.build_cnn(ck["backbone"], bool(ck.get("use_metadata", False)))
        net.load_state_dict(ck["state_dict"])
        net.eval().to(a.device)
        models.append(net)

    pre = P.ablation_to_pre(cfg["ablation"], img_size=cfg["img_size"],
                            input_norm=cfg["input_norm"], aug_plus=False)
    temp = None
    if src.get("calibration"):
        temp = np.asarray([float(src["calibration"]["temperature"][k]) for k in _SURFACES],
                          dtype=np.float32)

    items = []
    for it in src["items"]:
        gray = cv2.imdecode(np.frombuffer(base64.b64decode(it["crop_png_b64"]), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(it["mask_png_b64"]), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            print(f"[in-env ref] FAIL cannot decode crops of {it['name']}")
            return 1
        img = P.preprocess_tooth(gray, mask > 127, int(it["fdi"]), pre)
        x = np.stack([img, img, img], axis=0).astype(np.float32)
        if pre.input_norm == "imagenet":
            mu = np.asarray(cfg["imagenet_mu"], dtype=np.float32).reshape(3, 1, 1)
            sd = np.asarray(cfg["imagenet_sd"], dtype=np.float32).reshape(3, 1, 1)
            x = (x - mu) / sd
        with torch.no_grad():
            xb = torch.from_numpy(x[None]).to(a.device)
            probs = None
            for net in models:
                p = torch.sigmoid(net(xb)).cpu().numpy()
                probs = p if probs is None else probs + p
            probs = probs / float(len(models))
            if temp is not None:
                pc = np.clip(probs, 1e-6, 1 - 1e-6)
                z = np.log(pc / (1 - pc))
                probs = 1.0 / (1.0 + np.exp(-z / temp))
        items.append({
            "name": it["name"],
            "case_id": it["case_id"],
            "fdi": it["fdi"],
            "label": it.get("label"),
            "crop_png_b64": it["crop_png_b64"],
            "mask_png_b64": it["mask_png_b64"],
            "tensor_sha256": hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest(),
            "tensor_shape": list(x.shape),
            "grid_f16_b64": base64.b64encode(
                cv2.resize(x[0].astype(np.float32), (64, 64),
                           interpolation=cv2.INTER_AREA).astype(np.float16).tobytes()
            ).decode("ascii"),
            "probs_uncalibrated": None,
            "probs": {k: float(probs[0][j]) for j, k in enumerate(_SURFACES)},
        })
        print(f"[in-env ref] {it['name']}: tensor={items[-1]['tensor_sha256'][:12]} "
              f"probs={[round(float(v), 6) for v in probs[0]]}")

    payload = dict(src)
    payload["note"] = ("reference REBUILT in this environment by the ORIGINAL pipeline.py "
                       f"(crops reused from {a.src})")
    payload["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "pipeline_dir": root,
    }
    payload["items"] = items

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[in-env ref] wrote {dst} ({dst.stat().st_size / 1024:.0f} KiB) "
          f"items={len(items)} checkpoints={len(models)} "
          f"env={payload['environment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
