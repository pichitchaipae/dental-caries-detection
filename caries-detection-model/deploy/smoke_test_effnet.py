"""HTTP smoke test for the staging EfficientNet surface service.

What it proves
--------------
* ``/health`` reports the loaded folds and artifacts.
* The exact fold test-half (same case split as ``train_fold``) can be scored
  through the HTTP contract with one fold model, and the resulting
  recall/precision/F1/PR-AUC reproduce ``best_pca_s42_f<fold>_metrics.json``.
* Per-tooth probabilities match ``_score_checkpoint`` (independent code path).
* Malformed requests are rejected with 422.

It never claims new performance: the samples are the fold's internal test half
and the numbers are compared with the already-reported ones.

Run (service already listening):
  python deploy/smoke_test_effnet.py --url http://127.0.0.1:8011 --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # keep the training GPU untouched

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402

SURFACES = list(P.SURFACES)


def _png(arr: np.ndarray) -> bytes:
    """PNG-encode an image for the multipart upload.

    Bool masks MUST become 0/255: cv2.imencode falls back to CV_8U and would
    otherwise write 0/1, which the service (thresholding at >127, like the
    training cache) would read as an empty mask.
    """
    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    ok, buf = cv2.imencode(".png", a)
    if not ok:  # pragma: no cover - defensive
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _read_folds(path: Path) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid, folds = line.split(",")
            out[int(cid)] = [int(v) for v in folds.split(";") if v]
    return out


def _post(url: str, items: list[tuple[np.ndarray, np.ndarray, int]],
          fold: int) -> dict:
    files = []
    for gray, mask, _fdi in items:
        files.append(("crops", ("crop.png", _png(gray), "image/png")))
        files.append(("masks", ("mask.png", _png(mask), "image/png")))
    data = {"fdis": ",".join(str(f) for _g, _m, f in items),
            "fold": str(fold)}
    r = requests.post(url.rstrip("/") + "/v1/surface/predict",
                      files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json()


def _metrics(y: np.ndarray, p: np.ndarray, thresholds) -> dict:
    return P.compute_metrics(y, p, thresholds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--parity-n", type=int, default=60)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = P.OUTPUTS_DIR / "deploy"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = Path(args.out) if args.out else (
        outdir / f"smoke_f{args.fold}_{stamp}.json")
    evidence: dict = {"timestamp": stamp, "url": args.url,
                      "fold": args.fold, "status": "RUNNING"}

    health = requests.get(args.url.rstrip("/") + "/health", timeout=60)
    health.raise_for_status()
    evidence["health"] = health.json()
    assert evidence["health"]["ready"], evidence["health"].get("errors")
    print(f"[smoke] health ready; folds={evidence['health']['n_folds']}")

    ckpt = P.OUTPUTS_DIR / f"best_pca_s42_f{args.fold}.pt"
    assert ckpt.is_file(), f"missing {ckpt}"
    stored = json.loads(
        (P.OUTPUTS_DIR / f"best_pca_s42_f{args.fold}_metrics.json")
        .read_text(encoding="utf-8"))
    thresholds = [stored["thresholds"][SURFACES.index(s)] for s in SURFACES]

    _meta, samples = P.load_manifest()
    samples, _n_exc = P.apply_exclude(
        samples, "@" + str(P.OUTPUTS_DIR / "peeked_cases.json"))
    folds = _read_folds(P.OUTPUTS_DIR /
                        "folds_pca_stratified-groupkfold_seed42_exc100.csv")
    va = [s for s in samples if args.fold in folds.get(int(s["case_id"]), [])]
    _tune, test = P._tune_test_split(va, 42 + 7919)
    print(f"[smoke] fold {args.fold}: test teeth={len(test)} "
          f"cases={len({s['case_id'] for s in test})}")

    # ---- full test-half through HTTP (single fold model) -------------------
    y = np.stack([s["label"] for s in test]).astype(int)
    probs = []
    t0 = time.perf_counter()
    for i in range(0, len(test), args.batch):
        chunk = test[i:i + args.batch]
        items = []
        for s in chunk:
            gray = cv2.imread(str(P.CACHE_DIR / s["crop_path"]),
                              cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(P.CACHE_DIR / s["mask_path"]),
                              cv2.IMREAD_GRAYSCALE)
            assert gray is not None and mask is not None
            items.append((gray, mask > 127, int(s["fdi"])))
        resp = _post(args.url, items, args.fold)
        probs.extend([p["probabilities"] for p in resp["predictions"]])
        if (i // args.batch) % 10 == 0:
            print(f"[smoke] scored {min(i + args.batch, len(test))}/{len(test)}")
    probs = np.asarray(probs, dtype=np.float64)
    wall = time.perf_counter() - t0
    assert probs.shape == y.shape, (probs.shape, y.shape)

    got = _metrics(y, probs, thresholds)
    diffs = {}
    for s in SURFACES + ["HAS_CARIES(any)"]:
        diffs[s] = {k: abs(float(got[s][k]) - float(stored["test"][s][k]))
                    for k in ("recall", "precision", "f1", "pr_auc")}
    max_metric_diff = max(v for d in diffs.values() for v in d.values())
    evidence["test_half"] = {
        "n_teeth": int(len(test)),
        "n_cases": len({s["case_id"] for s in test}),
        "wall_seconds": round(wall, 2),
        "max_abs_metric_diff_vs_stored": max_metric_diff,
        "per_surface": {s: {"service": got[s], "stored": stored["test"][s],
                            "abs_diff": diffs[s]}
                        for s in SURFACES + ["HAS_CARIES(any)"]},
    }
    print(f"[smoke] metric parity max|diff| = {max_metric_diff:.3e}")
    # The stored metrics were produced on CUDA; this staging service runs on
    # CPU. Cross-device float32 noise moves PR-AUC by ~6e-4 (measured on the
    # reference path), so 2e-3 is the correct reproduction tolerance here.
    assert max_metric_diff < 2e-3, "service metrics do not reproduce the fold"

    # ---- per-tooth parity vs _score_checkpoint -----------------------------
    rng = np.random.RandomState(123)
    idx = rng.choice(len(test), size=min(args.parity_n, len(test)),
                     replace=False)
    subset = [test[i] for i in idx]
    import torch
    torch.cuda.is_available = lambda: False  # reference path on CPU
    ref_logits = P._score_checkpoint(ckpt, subset, seed=42, num_workers=0)
    ref_p = 1.0 / (1.0 + np.exp(-ref_logits))
    items = []
    for s in subset:
        gray = cv2.imread(str(P.CACHE_DIR / s["crop_path"]),
                          cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(P.CACHE_DIR / s["mask_path"]),
                          cv2.IMREAD_GRAYSCALE)
        items.append((gray, mask > 127, int(s["fdi"])))
    resp = _post(args.url, items, args.fold)
    svc_p = np.asarray([p["probabilities"] for p in resp["predictions"]])
    max_prob_diff = float(np.abs(ref_p - svc_p).max())
    evidence["tooth_parity"] = {"n": len(subset),
                                "max_abs_prob_diff": max_prob_diff}
    print(f"[smoke] per-tooth parity max|diff| = {max_prob_diff:.3e}")
    assert max_prob_diff < 1e-5, "HTTP probabilities differ from reference"

    # ---- error handling ----------------------------------------------------
    one = items[:1]
    files = [("crops", ("crop.png", _png(one[0][0]), "image/png")),
             ("masks", ("m1.png", _png(one[0][1]), "image/png")),
             ("masks", ("m2.png", _png(one[0][1]), "image/png"))]
    bad = requests.post(args.url.rstrip("/") + "/v1/surface/predict",
                        files=files, data={"fdis": str(one[0][2])}, timeout=60)
    evidence["error_handling"] = {"mismatched_request_status": bad.status_code,
                                  "pass": bad.status_code == 422}
    print(f"[smoke] mismatched request -> {bad.status_code}")
    assert bad.status_code == 422

    evidence["status"] = "PASS"
    outpath.write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    print(f"[smoke] PASS -> {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
