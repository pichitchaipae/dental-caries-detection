"""Direct-run parity test: the staging serving path == the training path.

Checks, with no GPU and no HTTP server:
  1. The service's preprocessing, applied to a PNG round-trip of the cached
     crop/mask, equals ``ToothCropSet.__getitem__`` tensor exactly (max diff 0).
  2. The service's probabilities equal ``_score_checkpoint`` (same checkpoint)
     within float tolerance.

Run:  python tests/test_serve_parity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the training GPU

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402
from deploy_effnet_service import SurfaceEnsemble  # noqa: E402

CKPT = P.OUTPUTS_DIR / "best_pca_s42_f0.pt"


def main() -> int:
    if not CKPT.is_file():
        print(f"SKIP: {CKPT} not found yet")
        return 0
    svc = SurfaceEnsemble([CKPT], device="cpu")
    assert svc.ready, f"service not ready: {svc.errors}"

    _meta, samples = P.load_manifest()
    samples = [s for s in samples
               if (P.CACHE_DIR / s["crop_path"]).is_file()][:6]
    assert len(samples) == 6, "cache missing"

    max_t = 0.0
    for s in samples:
        gray = cv2.imread(str(P.CACHE_DIR / s["crop_path"]),
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(P.CACHE_DIR / s["mask_path"]),
                          cv2.IMREAD_GRAYSCALE)
        # Simulate the HTTP upload: PNG is lossless, so decoding must be exact.
        gray2 = cv2.imdecode(cv2.imencode(".png", gray)[1], cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imdecode(cv2.imencode(".png", mask)[1], cv2.IMREAD_GRAYSCALE)
        xt = svc.preprocess_one(gray2, mask2 > 127, int(s["fdi"]))
        ds = P.torch_dataset([s], svc.pre, train=False, seed=42)[0][0].numpy()
        max_t = max(max_t, float(np.abs(xt - ds).max()))
    print(f"tensor parity max|diff| = {max_t:.3e}")
    assert max_t == 0.0, "preprocessing differs from ToothCropSet"

    # torch 2.5.1 on Windows can report cuda.is_available()=True while
    # device_count()=0 after CUDA_VISIBLE_DEVICES is cleared; force the
    # reference path to CPU so the comparison is apples-to-apples.
    import torch
    torch.cuda.is_available = lambda: False
    ref_logits = P._score_checkpoint(CKPT, samples, seed=42, num_workers=0)
    ref_p = 1.0 / (1.0 + np.exp(-ref_logits))
    items = []
    for s in samples:
        gray = cv2.imread(str(P.CACHE_DIR / s["crop_path"]),
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(P.CACHE_DIR / s["mask_path"]),
                          cv2.IMREAD_GRAYSCALE)
        items.append((gray, mask > 127, int(s["fdi"])))
    got_p = np.asarray([r["probabilities"] for r in svc.predict_batch(items)])
    max_p = float(np.abs(ref_p - got_p).max())
    print(f"probability parity max|diff| = {max_p:.3e}")
    assert max_p < 1e-5, "model forward differs from _score_checkpoint"

    print("SERVE-PARITY-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
