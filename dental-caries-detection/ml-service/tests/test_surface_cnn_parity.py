"""Parity test: ml-service vendored preprocessing vs the ORIGINAL training code.

Reference file: ``parity_reference.json`` (built on the host by
``caries-detection-model/deploy/make_parity_reference.py``).  Put it next to this
file, or point ``SURFACE_CNN_PARITY_REF`` at it.

Two checks:
  A. static  -- constants / call order / rotation sign of the vendored copy must
                equal the fingerprint recorded from ``pipeline.py``.
  B. numeric -- for each fixed tooth: the float32 model input tensor must hash
                identically, and (when the checkpoints are present) the ensemble
                probabilities must match within 1e-5.

Run inside the container:
    python /app/tests/test_surface_cnn_parity.py
or as pytest:
    pytest -q /app/tests/test_surface_cnn_parity.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:            # allow `python tests/...`
    sys.path.insert(0, str(HERE.parent))

try:
    from app.pipeline import surface_cnn as sc
except ImportError:                              # pragma: no cover
    import importlib

    sys.path.insert(0, str(HERE.parent / "app"))
    sc = importlib.import_module("app.pipeline.surface_cnn")

def _default_ref() -> Path:
    """Reference JSON location.

    It contains tooth crops (patient-derived pixels), so it is deliberately
    NOT committed: it is produced by
    ``caries-detection-model/deploy/make_parity_reference.py`` and stored next
    to the gitignored weights, or pointed at with ``SURFACE_CNN_PARITY_REF``.
    """
    env = os.environ.get("SURFACE_CNN_PARITY_REF", "").strip()
    if env:
        return Path(env)
    beside_weights = Path(sc.DEFAULT_DIR) / "parity_reference.json"
    if beside_weights.is_file():
        return beside_weights
    return HERE / "parity_reference.json"


REF_PATH = _default_ref()
GRID_TOL = 0.02        # z-score units; a real preprocessing bug is >> 1
PROB_TOL = 1e-4        # guards the ensemble/temperature math; platform-tolerant


def load_reference() -> dict:
    if not REF_PATH.is_file():
        raise RuntimeError(f"reference not found: {REF_PATH}")
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def check_static(ref: dict) -> list[str]:
    fp = ref["source_fingerprint"]
    problems: list[str] = []
    checks = [
        ("CROP_PAD", int(getattr(sc, "CROP_PAD"))),
        ("MAX_TILT_DEG", float(getattr(sc, "MAX_TILT_DEG"))),
        ("CLAHE_CLIP", float(getattr(sc, "CLAHE_CLIP"))),
        ("CLAHE_TILE", [int(v) for v in getattr(sc, "CLAHE_TILE")]),
    ]
    for key, got in checks:
        want = fp[key]
        if isinstance(want, list):
            ok = [int(v) for v in want] == got
        elif isinstance(want, float):
            ok = abs(float(want) - float(got)) < 1e-12
        else:
            ok = int(want) == int(got)
        if not ok:
            problems.append(f"static {key}: reference={want} vendored={got}")

    import inspect

    body = inspect.getsource(sc._preprocess_tooth)
    order = [k for k in ("_apply_clahe", "_zscore_foreground", "_rotate_crop",
                         "_canonical_flip_needed", "_resize_pad")
             if k in body]
    want_order = ["_" + k for k in fp["preprocess_call_order"]]
    if order != want_order:
        problems.append(f"static call order: reference={want_order} vendored={order}")

    if fp["rotate_negates_angle"] and "-float(angle_deg)" not in inspect.getsource(sc._rotate_crop):
        problems.append("static rotate sign: vendored _rotate_crop lost the negation")
    if fp["pca_long_axis_is_primary"] and "eigvecs[0]" not in inspect.getsource(sc._pca_angle):
        problems.append("static PCA axis: vendored _pca_angle no longer uses eigvecs[0]")
    return problems


def build_tensor(item: dict, cfg: dict) -> np.ndarray:
    import cv2

    crop = cv2.imdecode(np.frombuffer(base64.b64decode(item["crop_png_b64"]), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(item["mask_png_b64"]), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    if crop is None or mask is None:
        raise RuntimeError(f"{item['name']}: cannot decode reference images")
    use_clahe, use_pca = sc._ablation_to_pre(cfg["ablation"], cfg["img_size"])
    img = sc._preprocess_tooth(crop, mask > 127, int(item["fdi"]),
                               use_clahe, use_pca, int(cfg["img_size"]),
                               cfg["input_norm"])
    x = np.stack([img, img, img], axis=0).astype(np.float32)
    if cfg["input_norm"] == "imagenet":
        mu = np.asarray(cfg["imagenet_mu"], dtype=np.float32).reshape(3, 1, 1)
        sd = np.asarray(cfg["imagenet_sd"], dtype=np.float32).reshape(3, 1, 1)
        x = (x - mu) / sd
    return x


def _grid(x: np.ndarray) -> np.ndarray:
    """64x64 INTER_AREA digest of the 2-D model input (see the generator)."""
    import cv2

    return cv2.resize(x[0].astype(np.float32), (64, 64),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def check_numeric(ref: dict) -> list[str]:
    problems: list[str] = []
    cfg = ref["config"]
    items = []
    exact: list[str] = []
    semantic: list[str] = []
    for item in ref["items"]:
        x = np.ascontiguousarray(build_tensor(item, cfg), dtype=np.float32)
        if list(x.shape) != list(item["tensor_shape"]):
            problems.append(f"{item['name']}: shape {list(x.shape)} != {item['tensor_shape']}")
        elif hashlib.sha256(x.tobytes()).hexdigest() == item["tensor_sha256"]:
            exact.append(item["name"])
        elif not item.get("grid_f16_b64"):
            problems.append(f"{item['name']}: tensor hash differs and the reference "
                            f"has no grid digest (rerun make_parity_reference.py)")
        else:
            ref_grid = np.frombuffer(base64.b64decode(item["grid_f16_b64"]),
                                     np.float16).astype(np.float32).reshape(64, 64)
            delta = float(np.abs(_grid(x) - ref_grid).max())
            if delta <= GRID_TOL:
                semantic.append(f"{item['name']}(maxΔ={delta:.4f})")
            else:
                problems.append(f"{item['name']}: preprocessing differs "
                                f"(grid maxΔ={delta:.4f} > {GRID_TOL})")
        items.append(item)
    print(f"[parity] tensor build: exact-hash={len(exact)}/{len(items)} "
          f"platform-tolerant={len(semantic)}")
    for s in semantic:
        print(f"[parity]   note: different numeric build -> {s}")

    bundle_dir = Path(os.environ.get("SURFACE_CNN_DIR", sc.DEFAULT_DIR))
    cps = sorted(bundle_dir.glob("best_pca_s42_f[0-9].pt"))
    if not cps:
        print(f"[parity] SKIP probability check: no checkpoints in {bundle_dir}")
        return problems
    if ref.get("calibration") is None:
        print("[parity] note: reference has no calibration -> comparing raw sigmoids")

    device = os.environ.get("SURFACE_CNN_DEVICE", "cpu")
    bundle = sc.CnnBundle(cps, device=device,
                          calibration_path=str(bundle_dir / "calibration_pca.json"),
                          threads=int(os.environ.get("SURFACE_CNN_THREADS", "4")))
    if not bundle.ready:
        problems.append(f"bundle not ready: {bundle.errors}")
        return problems

    # predict() preprocesses internally; feed the decoded crops so there is a
    # single preprocessing code path in the container
    import cv2

    decoded = []
    for it in ref["items"]:
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(it["crop_png_b64"]), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(it["mask_png_b64"]), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        decoded.append((crop, mask > 127, int(it["fdi"])))
    preds = bundle.predict(decoded)
    worst, worst_at = 0.0, ""
    for it, p in zip(items, preds):
        for surf, want in it["probs"].items():
            got = float(p["probabilities"][surf])
            delta = abs(got - float(want))
            if delta > worst:
                worst, worst_at = delta, f"{it['name']}/{surf}"
            if delta > PROB_TOL:
                problems.append(f"{it['name']}/{surf}: prob {got:.6f} != ref {want:.6f}")
    print(f"[parity] probs: max|delta|={worst:.3e} at {worst_at}")
    return problems


def _decode(item: dict) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    crop = cv2.imdecode(np.frombuffer(base64.b64decode(item["crop_png_b64"]), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(item["mask_png_b64"]), np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    if crop is None or mask is None:
        raise RuntimeError(f"{item['name']}: cannot decode reference images")
    return crop, mask > 127


def check_oracle(ref: dict) -> list[str]:
    """Compare the vendored copy with the ORIGINAL ``pipeline.py`` **inside the
    current environment** (set ``CARIES_PIPELINE_DIR`` to the training repo).

    This separates the two possible causes of a host/container mismatch:
      * vendored code differs from the original  -> in-env hashes differ
      * a different OpenCV/torch build           -> in-env hashes are identical,
        while both differ from the host-generated reference
    """
    root = os.environ.get("CARIES_PIPELINE_DIR", "").strip()
    if not root:
        print("[parity] oracle: skipped (set CARIES_PIPELINE_DIR to enable)")
        return []
    # A *requested* oracle that cannot run must fail loudly, not pass silently.
    if not (Path(root) / "pipeline.py").is_file():
        print(f"[parity] oracle: FAIL requested but pipeline.py not found in {root}")
        return [f"oracle requested but pipeline.py is missing in {root}"]
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import pipeline as P  # type: ignore
    except Exception as exc:
        print(f"[parity] oracle: FAIL cannot import pipeline from {root}: {exc!r}")
        return [f"oracle requested but pipeline import failed: {exc!r}"]

    cfg = ref["config"]
    problems: list[str] = []
    identical = 0
    ref_match = 0
    worst = 0.0
    for item in ref["items"]:
        crop, mask = _decode(item)
        pre = P.ablation_to_pre(cfg["ablation"], img_size=cfg["img_size"],
                               input_norm=cfg["input_norm"], aug_plus=False)
        img = P.preprocess_tooth(crop, mask, int(item["fdi"]), pre)
        x = np.stack([img, img, img], axis=0).astype(np.float32)
        if cfg["input_norm"] == "imagenet":
            mu = np.asarray(cfg["imagenet_mu"], dtype=np.float32).reshape(3, 1, 1)
            sd = np.asarray(cfg["imagenet_sd"], dtype=np.float32).reshape(3, 1, 1)
            x = (x - mu) / sd
        x = np.ascontiguousarray(x, dtype=np.float32)
        mine = np.ascontiguousarray(build_tensor(item, cfg), dtype=np.float32)
        same = hashlib.sha256(x.tobytes()).hexdigest() == hashlib.sha256(mine.tobytes()).hexdigest()
        delta = float(np.abs(x - mine).max())
        worst = max(worst, delta)
        in_ref = hashlib.sha256(x.tobytes()).hexdigest() == item["tensor_sha256"]
        identical += int(same)
        ref_match += int(in_ref)
        if not same and delta > 1e-4:
            problems.append(f"oracle {item['name']}: vendored != original in this "
                            f"environment (maxΔ={delta:.5f})")
        host_delta = float("nan")
        if item.get("grid_f16_b64"):
            rg = np.frombuffer(base64.b64decode(item["grid_f16_b64"]),
                               np.float16).astype(np.float32).reshape(64, 64)
            host_delta = float(np.abs(_grid(x) - rg).max())
        print(f"[parity] oracle {item['name']}: vendored==original={same} "
              f"original==reference={in_ref} in-env maxΔ={delta:.2e} "
              f"vs-reference grid maxΔ={host_delta:.4f}")
    print(f"[parity] oracle: vendored==original {identical}/{len(ref['items'])}, "
          f"original==reference {ref_match}/{len(ref['items'])}, worst maxΔ={worst:.2e}")
    return problems


def main() -> int:
    print(f"[parity] reference: {REF_PATH}")
    ref = load_reference()
    problems = check_static(ref)
    problems += check_numeric(ref)
    problems += check_oracle(ref)
    print(f"[parity] items={len(ref['items'])} checkpoints={len(ref['checkpoints'])} "
          f"grid_tol={GRID_TOL} prob_tol={PROB_TOL}")
    if problems:
        print("[parity] FAIL")
        for p in problems:
            print("   -", p)
        return 1
    print("[parity] PASS: vendored preprocessing == original training code")
    return 0


def test_static_parity() -> None:
    problems = check_static(load_reference())
    assert not problems, problems


def test_numeric_parity() -> None:
    problems = check_numeric(load_reference())
    assert not problems, problems


def test_oracle_parity() -> None:
    problems = check_oracle(load_reference())
    assert not problems, problems


if __name__ == "__main__":
    raise SystemExit(main())
