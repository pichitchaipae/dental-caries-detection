"""Benchmark: CPU latency + memory of the 5-fold CNN surface bundle.

Measures the numbers required by the Stage-3 shadow checklist:
  * model load time,
  * per-tooth latency for one tooth (call overhead) and for a full pano batch,
  * RSS memory before load / after load / after a batch.

The teeth come from ``parity_reference.json`` (the same six cached crops used
for the parity test), repeated up to ``--teeth`` so a full-mouth workload is
simulated.  That file lives in the gitignored weights directory: it contains
patient-derived crops and must never be committed.

Run (container, and/or host):
    SURFACE_CNN_DIR=/weights/cnn_pca42 python /app/tests/bench_surface_cnn.py --teeth 30

There is deliberately NO pass/fail gate here -- these are reported engineering
numbers, not a clinical acceptance criterion.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:            # allow `python tests/...`
    sys.path.insert(0, str(HERE.parent))

from app.pipeline import surface_cnn as sc           # noqa: E402


def rss_mb() -> float | None:
    """Resident set size in MiB (Linux /proc, then resource fallback)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / 1024.0 if platform.system() != "Darwin" else rss / (1024.0 * 1024.0)
    except Exception:                            # pragma: no cover - defensive
        return None


def load_items(ref_path: Path, n: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    import cv2

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    base: list[tuple[np.ndarray, np.ndarray, int]] = []
    for it in ref["items"]:
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(it["crop_png_b64"]), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(it["mask_png_b64"]), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if crop is None or mask is None:
            continue
        base.append((crop, mask > 127, int(it["fdi"])))
    if not base:
        raise RuntimeError(f"no usable items in {ref_path}")
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    while len(out) < n:
        out.extend(base)
    return out[:n]


def versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch
        import torchvision

        out["torch"] = torch.__version__
        out["torchvision"] = torchvision.__version__
    except Exception:                            # pragma: no cover
        pass
    try:
        import cv2

        out["opencv"] = cv2.__version__
    except Exception:                            # pragma: no cover
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get("SURFACE_CNN_DIR", sc.DEFAULT_DIR),
                    help="folder with the 5 fold checkpoints + calibration")
    ap.add_argument("--ref", default=os.environ.get("SURFACE_CNN_PARITY_REF") or None,
                    help="parity_reference.json (default: <dir>/parity_reference.json, "
                         "or $SURFACE_CNN_PARITY_REF)")
    ap.add_argument("--teeth", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--device", default=os.environ.get("SURFACE_CNN_DEVICE", "cpu"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SURFACE_CNN_THREADS", "4")))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = Path(a.dir)
    ref_path = Path(a.ref) if a.ref else d / "parity_reference.json"
    cps = sorted(d.glob("best_pca_s42_f[0-9].pt"))
    if not cps:
        print(f"[bench] FAIL no checkpoints in {d}")
        return 1
    if not ref_path.is_file():
        print(f"[bench] FAIL reference not found: {ref_path}")
        return 1

    m_before = rss_mb()
    t0 = time.perf_counter()
    bundle = sc.CnnBundle(cps, device=a.device,
                          calibration_path=str(d / "calibration_pca.json"),
                          threads=a.threads)
    load_s = time.perf_counter() - t0
    m_loaded = rss_mb()
    if not bundle.ready:
        print(f"[bench] FAIL bundle not ready: {bundle.errors}")
        return 1

    items = load_items(ref_path, max(1, a.teeth))
    for _ in range(a.warmup):                    # warm the threads/allocator
        bundle.predict(items[:1])

    t0 = time.perf_counter()
    bundle.predict(items[:1])
    one_ms = (time.perf_counter() - t0) * 1000.0

    batch_ms: list[float] = []
    for _ in range(max(1, a.repeat)):
        t0 = time.perf_counter()
        bundle.predict(items)
        batch_ms.append((time.perf_counter() - t0) * 1000.0)
    m_after = rss_mb()

    report = {
        "folder": str(d),
        "checkpoints": [p.name for p in cps],
        "n_folds": len(bundle.models),
        "device": a.device,
        "threads": a.threads,
        "teeth": len(items),
        "repeat": max(1, a.repeat),
        "load_s": round(load_s, 3),
        "single_tooth_ms": round(one_ms, 1),
        "batch_ms": [round(v, 1) for v in batch_ms],
        "batch_median_ms": round(statistics.median(batch_ms), 1),
        "batch_per_tooth_ms": round(statistics.median(batch_ms) / len(items), 2),
        "rss_mb": {"before_load": m_before, "after_load": m_loaded, "after_batch": m_after},
        "versions": versions(),
        "note": "engineering numbers only (CPU, no clinical acceptance implied)",
    }
    print("[bench] " + json.dumps(report, ensure_ascii=False))
    print(f"[bench] load={report['load_s']}s 1-tooth={report['single_tooth_ms']}ms "
          f"batch({len(items)})={report['batch_median_ms']}ms "
          f"per-tooth={report['batch_per_tooth_ms']}ms "
          f"rss={m_before}->{m_loaded}->{m_after} MiB")
    default_out = (Path(os.environ["BENCH_OUT"]) if os.environ.get("BENCH_OUT")
                   else Path(os.environ.get("SHARED_DIR", "/tmp")) / "surface_cnn_bench.json")
    out = Path(a.out) if a.out else default_out
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[bench] wrote {out}")
    except OSError as exc:
        # /weights is mounted read-only at runtime; the JSON line above is the report.
        print(f"[bench] note: could not write {out} ({exc}); use the JSON line above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
