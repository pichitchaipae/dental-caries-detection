"""Step 4 — offline end-to-end evaluation of the surface-CNN shadow path.

STATUS: REFERENCE ONLY.  The CNN surface path is not part of serving any more;
production surfaces come from ``app/pipeline/surface_heuristic.py``.  This
script is kept to reproduce the historical CNN shadow evaluation and is not
collected by pytest.  It therefore runs Stage 1 with the caries model disabled
(``caries=None``) and does not require a ``caries_detector`` entry in
``models/versions.py``.

Declared BEFORE the run (reviewer-approved; nothing here may be re-tuned):

  1. set          : every tooth of the OOF CSV (exc100) — 201 cases / 6,012 teeth
  2. baseline     : ``prob_*`` / ``pred_*`` columns of that CSV = cached training
                    crops, each case scored by the fold model that never saw it
  3. e2e path     : panoramic PNG -> ml-service stage 1 (DETECTION_THRESHOLD=0.25,
                    ENABLE_CROP_SEGMENTER=false) -> polygon crop + CROP_PAD ->
                    the SAME held-out fold model -> RAW sigmoid probabilities
                    (the OOF CSV is uncalibrated: pipeline.py stores ``evaluate()``
                    outputs, temperature is only fitted later for the peeked-set
                    calibration) -> decisions at the fold checkpoint's stored
                    thresholds (no re-tuning)
  4. matching     : ``(case_id, fdi)``; several detections with one FDI in a case
                    -> per-surface maximum probability
  5. not detected : probability 0 for every surface (counted as a miss; coverage
                    is reported separately)
  6. extra teeth  : dropped (no ground truth)
  7. metrics      : AP     = fold-mean ``average_precision_score`` (identical to
                    ``pipeline.oof_per_fold``), nan for degenerate folds;
                    Recall = pooled at the stored thresholds
  8. self-check   : must reproduce the published OOF report first, otherwise the
                    script stops without running the evaluation
  9. gate         : dAP <= 0.05 AND dRecall <= 0.05 per surface (engineering gate
                    only, not clinical acceptance)

Runs INSIDE the ml-service container (stage 1 needs ultralytics + pinned weights);
images are staged as ``<images-root>/case N/case_N.png``:

    python /tmp/psd/e2e_offline_eval.py \
        --cases-csv /tmp/psd/oof_pca_stratified-groupkfold_seed42_exc100.csv \
        --report    /tmp/psd/oof_report_oof_pca_stratified-groupkfold_seed42_exc100.json \
        --images-root /data \
        --weights /weights --folds-dir /weights/cnn_pca42 \
        --out /tmp/e2e_eval.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# The harness runs inside the ml-service container, where the service package
# lives at /app.  "python /tmp/psd/e2e_offline_eval.py" puts /tmp/psd (the
# script directory), not the working directory, on sys.path, so add the service
# root explicitly (override with ML_SERVICE_ROOT if the layout ever changes).
_SERVICE_ROOT = os.environ.get("ML_SERVICE_ROOT", "/app")
if _SERVICE_ROOT and _SERVICE_ROOT not in sys.path:
    sys.path.insert(0, _SERVICE_ROOT)

import numpy as np

SURFACES = ["Distal", "Mesial", "Occlusal"]
GATE = 0.05


# ---------------------------------------------------------------------------
# OOF CSV / metrics (mirrors pipeline.load_oof_csv + pipeline.oof_per_fold)
# ---------------------------------------------------------------------------

def read_oof(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"empty OOF csv: {path}")
    out = []
    for r in rows:
        out.append({
            "case": int(r["case_id"]),
            "fdi": int(r["tooth_fdi"]),
            "fold": int(float(r["fold"])),
            "y": [int(float(r["true_" + s])) for s in SURFACES],
            "p": [float(r["prob_" + s]) for s in SURFACES],
            "d": [int(float(r["pred_" + s])) for s in SURFACES],
        })
    return out


def ap_fold_mean(rows: list[dict], values: dict[int, list[float]]) -> tuple[dict, dict]:
    """values: row-index -> 3 probabilities.  Returns (mean, per-fold)."""
    from sklearn.metrics import average_precision_score

    folds = sorted({r["fold"] for r in rows})
    per_fold: dict[str, list[float]] = {s: [] for s in SURFACES}
    for f in folds:
        idx = [i for i, r in enumerate(rows) if r["fold"] == f]
        for j, s in enumerate(SURFACES):
            y = np.array([rows[i]["y"][j] for i in idx])
            p = np.array([values[i][j] for i in idx])
            if y.sum() == 0 or y.sum() == len(y):
                per_fold[s].append(float("nan"))
            else:
                per_fold[s].append(float(average_precision_score(y, p)))
    return {s: float(np.nanmean(per_fold[s])) for s in SURFACES}, per_fold


def pooled(rows: list[dict], preds: dict[int, list[int]]) -> dict:
    out = {}
    for j, s in enumerate(SURFACES):
        y = np.array([r["y"][j] for r in rows])
        d = np.array([preds[i][j] for i in range(len(rows))])
        tp = int(((y == 1) & (d == 1)).sum())
        fp = int(((y == 0) & (d == 1)).sum())
        fn = int(((y == 1) & (d == 0)).sum())
        out[s] = {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                  "tp": tp, "fp": fp, "fn": fn}
    return out


def fold_thresholds(folds_dir: Path) -> dict[int, dict[str, float]]:
    """Stored per-fold thresholds (tuned on the tune-half, never on the test-half)."""
    import torch

    out: dict[int, dict[str, float]] = {}
    for cp in sorted(folds_dir.glob("best_pca_s42_f[0-9].pt")):
        fold = int(cp.stem.rsplit("_f", 1)[1])
        ck = torch.load(str(cp), map_location="cpu", weights_only=False)
        order = list(ck["surfaces"])
        thr = [float(v) for v in ck["thresholds"]]
        out[fold] = {s: thr[order.index(s)] for s in SURFACES}
    if not out:
        raise SystemExit(f"no checkpoints in {folds_dir}")
    return out


# ---------------------------------------------------------------------------
# Self-check — must reproduce the published OOF report
# ---------------------------------------------------------------------------

def self_check(rows: list[dict], report: dict, thr: dict[int, dict[str, float]]) -> bool:
    ok = True

    def want(s: str, key: str) -> float | None:
        try:
            return float(report["per_surface"][s][key])
        except Exception:
            return None

    def verdict(good: bool, w: float | None) -> str:
        return "OK" if good else ("MISMATCH" if w is not None else "MISSING")

    base_p = {i: r["p"] for i, r in enumerate(rows)}
    ap_mean, _ = ap_fold_mean(rows, base_p)
    for s in SURFACES:
        w, got = want(s, "ap_fold_mean"), ap_mean[s]
        good = w is not None and abs(got - w) <= 1e-6
        ok &= good
        print(f"[self-check] S1 {s:<8} fold-mean AP {got:.6f} "
              f"vs report {w if w is None else round(w, 6)} -> {verdict(good, w)}",
              flush=True)

    # S2: the CSV's own pred columns must follow from its prob columns + the
    # stored per-fold thresholds (proves which thresholds the protocol used).
    mism, boundary = 0, 0
    for i, r in enumerate(rows):
        t = thr[r["fold"]]
        for j, s in enumerate(SURFACES):
            pred = int(r["p"][j] >= t[s])
            if pred != r["d"][j]:
                mism += 1
                if abs(r["p"][j] - t[s]) <= 1e-3:      # 4-decimal CSV rounding band
                    boundary += 1
    good = mism == 0 or mism == boundary
    ok &= good
    print(f"[self-check] S2 pred recompute: mismatches={mism} "
          f"(of which rounding-boundary={boundary}) -> {'OK' if good else 'MISMATCH'}",
          flush=True)

    base_d = {i: r["d"] for i, r in enumerate(rows)}
    pool = pooled(rows, base_d)
    for s in SURFACES:
        w, got = want(s, "recall_pooled"), pool[s]["recall"]
        good = w is not None and abs(got - w) <= 1e-6
        ok &= good
        print(f"[self-check] S3 {s:<8} pooled recall {got:.6f} "
              f"vs report {w if w is None else round(w, 6)} -> {verdict(good, w)}",
              flush=True)

    print(f"[self-check] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ---------------------------------------------------------------------------
# End-to-end pass
# ---------------------------------------------------------------------------

def run_e2e(args, rows: list[dict]) -> dict:
    # config.py builds Settings() at import time and DATABASE_URL is required
    os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/e2e_eval.db")
    os.environ["WEIGHTS_DIR"] = str(args.weights)
    os.environ["ENABLE_CROP_SEGMENTER"] = "false"      # serving intent
    os.environ.setdefault("ML_DEVICE", "cpu")
    os.environ.setdefault("DETECTION_THRESHOLD", "0.25")

    from app.pipeline.detection import run_stage1
    from app.pipeline.surface_cnn import CnnBundle, build_items
    from models.versions import MODEL_VERSIONS
    from ultralytics import YOLO

    thr = fold_thresholds(Path(args.folds_dir))
    print(f"[e2e] thresholds per fold: {json.dumps(thr, default=float)}", flush=True)

    # Reference run only: pano YOLO + the CNN checkpoints.  Crop-Seg and caries
    # are disabled so this historical evaluation does not depend on artifacts
    # that are optional or retired.
    w = Path(args.weights)
    models = {
        "pano": YOLO(str(w / MODEL_VERSIONS["pano_detector"]["filename"])),
        "caries": None,
        "crop": None,
    }
    bundles = {
        f: CnnBundle([Path(args.folds_dir) / f"best_pca_s42_f{f}.pt"], device="cpu",
                     calibration_path=None, threads=args.threads)
        for f in sorted(thr)
    }
    for f, b in bundles.items():
        if not b.ready:
            raise SystemExit(f"bundle fold {f} not ready: {b.errors}")

    cases: dict[int, dict] = {}
    order = sorted({r["case"] for r in rows})
    if args.limit:
        order = order[:args.limit]
    t_start = time.perf_counter()
    for n, cid in enumerate(order, 1):
        t0 = time.perf_counter()
        fold = next(r["fold"] for r in rows if r["case"] == cid)
        img_path = Path(args.images_root) / f"case {cid}" / f"case_{cid}.png"
        rec: dict = {"fold": fold, "image": str(img_path), "teeth": {}, "notes": []}
        if not img_path.is_file():
            rec["error"] = "image not found"
            cases[cid] = rec
            print(f"[e2e] case {cid} ({n}/{len(order)}): MISSING IMAGE", flush=True)
            continue
        try:
            detections, image = run_stage1(
                image_path=str(img_path), models=models,
                detection_threshold=0.25)
            items, notes = build_items(image, detections)
            preds = bundles[fold].predict(items) if items else []
            per: dict[int, list[list[float]]] = {}
            for p in preds:
                probs = [float(p["probabilities"][s]) for s in SURFACES]
                per.setdefault(int(p["fdi"]), []).append(probs)
            rec["n_detections"] = len(detections)
            rec["n_scored"] = len(preds)
            rec["notes"] = notes
            for fdi, group in per.items():
                rec["teeth"][str(fdi)] = {
                    "n_candidates": len(group),
                    "probabilities": {s: max(g[j] for g in group)
                                      for j, s in enumerate(SURFACES)},
                }
        except Exception as exc:  # keep going; the case counts as fully missed
            rec["error"] = repr(exc)
            print(f"[e2e] case {cid}: ERROR {exc!r}", flush=True)
        cases[cid] = rec
        if n % 10 == 0 or n == len(order):
            el = time.perf_counter() - t_start
            print(f"[e2e] {n}/{len(order)} cases  ({el:.0f}s, "
                  f"{el / n:.1f}s/case)", flush=True)

    if args.limit:
        all_p = [v for rec in cases.values() for t in rec.get("teeth", {}).values()
                 for v in t["probabilities"].values()]
        for cid in order:
            rec = cases[cid]
            print(f"[e2e] SMOKE case {cid}: fold={rec.get('fold')} "
                  f"det={rec.get('n_detections')} scored={rec.get('n_scored')} "
                  f"teeth={len(rec.get('teeth', {}))} "
                  f"notes={len(rec.get('notes', []))} error={rec.get('error')}",
                  flush=True)
        print(f"[e2e] SMOKE prob range: "
              f"{(min(all_p) if all_p else float('nan')):.6f} .. "
              f"{(max(all_p) if all_p else float('nan')):.6f} ({len(all_p)} values)",
              flush=True)
        return {"cases": cases, "smoke": True, "cases_evaluated": len(order),
                "wall_s": round(time.perf_counter() - t_start, 1)}

    # ---- score against the declared set ------------------------------------
    e2e_p: dict[int, list[float]] = {}
    e2e_d: dict[int, list[int]] = {}
    found = 0
    for i, r in enumerate(rows):
        cid, fdi = r["case"], r["fdi"]
        rec = cases.get(cid, {})
        t = rec.get("teeth", {}).get(str(fdi))
        if t is None:
            e2e_p[i] = [0.0, 0.0, 0.0]
            e2e_d[i] = [0, 0, 0]
            continue
        found += 1
        probs = [t["probabilities"][s] for s in SURFACES]
        tf = thr[r["fold"]]
        e2e_p[i] = probs
        e2e_d[i] = [int(probs[j] >= tf[s]) for j, s in enumerate(SURFACES)]

    base_p = {i: r["p"] for i, r in enumerate(rows)}
    base_d = {i: r["d"] for i, r in enumerate(rows)}
    ap_base, _ = ap_fold_mean(rows, base_p)
    ap_e2e, _ = ap_fold_mean(rows, e2e_p)
    pool_base = pooled(rows, base_d)
    pool_e2e = pooled(rows, e2e_d)

    verdict = {}
    for s in SURFACES:
        dap = ap_e2e[s] - ap_base[s]
        drec = pool_e2e[s]["recall"] - pool_base[s]["recall"]
        verdict[s] = {
            "ap_base": ap_base[s], "ap_e2e": ap_e2e[s], "d_ap": dap,
            "recall_base": pool_base[s]["recall"], "recall_e2e": pool_e2e[s]["recall"],
            "d_recall": drec,
            "pass": bool(abs(dap) <= GATE and abs(drec) <= GATE),
        }

    print("\n[gate] surface   AP base    AP e2e     dAP     "
          "R base   R e2e    dRecall  verdict", flush=True)
    for s in SURFACES:
        v = verdict[s]
        print(f"[gate] {s:<9} {v['ap_base']:.4f}  {v['ap_e2e']:.4f}  "
              f"{v['d_ap']:+.4f}  {v['recall_base']:.4f}  {v['recall_e2e']:.4f}  "
              f"{v['d_recall']:+.4f}  {'PASS' if v['pass'] else 'FAIL'}", flush=True)
    overall = all(v["pass"] for v in verdict.values())
    print(f"[gate] coverage: {found}/{len(rows)} declared teeth matched "
          f"({100.0 * found / len(rows):.1f}%)", flush=True)
    print(f"[gate] OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(gate |dAP|,|dRecall| <= {GATE})", flush=True)

    return {"cases": cases, "verdict": verdict, "overall_pass": overall,
            "coverage": {"matched": found, "declared": len(rows)},
            "cases_evaluated": len(order), "cases_total": len({r["case"] for r in rows}),
            "ap_base": ap_base, "ap_e2e": ap_e2e,
            "pool_base": pool_base, "pool_e2e": pool_e2e,
            "thresholds": thr, "gate": GATE,
            "wall_s": round(time.perf_counter() - t_start, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-csv", required=True)
    ap.add_argument("--report", required=True, help="published OOF report json")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--folds-dir", default="/weights/cnn_pca42")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SURFACE_CNN_THREADS", "4")))
    ap.add_argument("--limit", type=int, default=0, help="smoke run over N cases")
    ap.add_argument("--self-check-only", action="store_true")
    ap.add_argument("--out", default="/tmp/e2e_eval.json")
    a = ap.parse_args()

    rows = read_oof(Path(a.cases_csv))
    cases_n = len({r["case"] for r in rows})
    print(f"[e2e] declared set: {len(rows)} teeth / {cases_n} cases "
          f"/ folds={sorted({r['fold'] for r in rows})}", flush=True)
    if not Path(a.report).is_file():
        print(f"[e2e] STOP: report json not found: {a.report}", flush=True)
        return 2
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    thr = fold_thresholds(Path(a.folds_dir))
    if not self_check(rows, report, thr):
        print("[e2e] STOP: self-check failed — the evaluation was not started", flush=True)
        return 2
    if a.self_check_only:
        return 0

    payload = run_e2e(a, rows)
    out = Path(a.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"[e2e] wrote {out}", flush=True)
    except OSError as exc:
        print(f"[e2e] note: could not write {out} ({exc})", flush=True)
    return 0 if payload.get("overall_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
