# Staging deployment — Stage-2 EfficientNet surface classifier

This directory implements a **staging-only** inference service for the
per-tooth caries surface classifier. It is deliberately placed inside
`caries-detection-model` and does **not** modify the training pipeline or the
existing `dental-caries-detection/ml-service` (which still serves the old
Random Forest model).

## What is deployed

| Item | Value |
|---|---|
| Task | multi-label per-surface caries classification (Distal / Mesial / Occlusal) |
| Input | grayscale tooth crop + binary tooth mask + FDI number |
| Output | probability per surface; threshold/positive only after calibration |
| Model | EfficientNet-B0, full fine-tuning, Focal Loss, `ablation=pca`, 384 px |
| Weights | `outputs/best_pca_s42_f{0..4}.pt` (completed folds only) |
| Ensemble | mean of per-fold sigmoid probabilities (matches `calibrate`) |
| Calibration | **not available yet** — requires all five folds + `pipeline.py calibrate` |
| Status | **staging only; not clinically validated; external test pending** |

The service imports `pipeline.preprocess_tooth`, `pipeline.build_cnn` and
`pipeline.ablation_to_pre` directly. Serving therefore uses exactly the training
transform; `tests/test_serve_parity.py` proves tensor equality and probability
equality against the training code paths.

## Start

```powershell
powershell -ExecutionPolicy Bypass -File deploy\run_staging.ps1 -Port 8011 -Folds "0,1,2,3"
```

Health and model metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:8011/health | ConvertTo-Json -Depth 6
```

Run the HTTP smoke test (uses the fold's exact internal test half):

```powershell
& "D:\Users\jaopi\anaconda3\envs\sp_project\python.exe" deploy\smoke_test_effnet.py --url http://127.0.0.1:8011 --fold 0
```

Evidence is written to `outputs/deploy/smoke_f<fold>_<UTC>.json`. It contains the
health payload, the metric comparison against
`outputs/best_pca_s42_f<fold>_metrics.json`, per-tooth probability parity against
`_score_checkpoint`, and the 422 error-handling check.

## Evidence collected (2026-10-18, staging on CPU)

| Check | Result | Artifact |
|---|---|---|
| Training-path parity (`tests/test_serve_parity.py`) | `SERVE-PARITY-OK` — tensor max diff **0.000e+00**, probability max diff **3.7e-09** | `outputs/deploy/parity_20261018.txt` |
| HTTP smoke, fold 0 (1 fold, port 8011) | **PASS** — 1,225 teeth / 41 cases; recall/precision/F1 diff vs stored **exactly 0**; PR-AUC max diff **5.93e-04** (CPU-vs-CUDA float32 noise, measured on the reference path); per-tooth parity **2.98e-08**; malformed request → 422 | `outputs/deploy/smoke_f0_20261018T104703Z.json` |
| Ensemble service (folds 0–3, port 8012) | `ready=true`, `n_folds=4`, no errors; known-caries tooth (label Distal=1) → prob 0.719 with ensemble (0.983 fold-0 only); healthy tooth → 0.111; `positive` stays `null` without calibration | `outputs/deploy/ensemble_smoke.json` |

Notes: the service runs on CPU so it never competes with the training queue
for the GPU; PR-AUC reproduction tolerance is 2e-3 because the stored fold
metrics were produced on CUDA while staging scores on CPU.

## API contract

`POST /v1/surface/predict` (multipart/form-data)

| Field | Type | Meaning |
|---|---|---|
| `crops` | file[] | grayscale PNG tooth crops, one per tooth |
| `masks` | file[] | binary PNG tooth masks, same size as the crop |
| `fdis` | string | comma-separated FDI numbers, one per crop |
| `fold` | int (optional) | force one fold model (used by OOF evaluation) |

Response:

```json
{
  "predictions": [
    {"fdi": 16,
     "surfaces": {
       "Distal":  {"probability": 0.31, "threshold": null, "positive": null},
       "Mesial":  {"probability": 0.22, "threshold": null, "positive": null},
       "Occlusal":{"probability": 0.48, "threshold": null, "positive": null}},
     "probabilities": [0.31, 0.22, 0.48]}
  ],
  "latency_ms": 41.2,
  "model": {"n_folds_used": 4, "fold": null, "calibration": "not_calibrated",
            "staging_only": true}
}
```

`threshold` and `positive` stay `null` until a calibration JSON is supplied via
`EFFNET_CALIBRATION`. This enforces the protocol rule that deployment
thresholds come only from `pipeline.py calibrate`.

## Go / no-go for staging

Pass all of the following:

1. `GET /health` returns `ready=true` with the expected checkpoint SHA-256 list.
2. `tests/test_serve_parity.py` prints `SERVE-PARITY-OK`.
3. `deploy/smoke_test_effnet.py` prints `PASS` and its evidence shows
   `max_abs_metric_diff_vs_stored < 1e-4` and `max_abs_prob_diff < 1e-5`.
4. The staging process log has no traceback.

Staging **does not** establish clinical readiness. External validation
(≥ ~103 blind-labeled cases), expert review (48 teeth, Cohen's kappa), and the
calibrated ensemble remain mandatory before any clinical claim.

## Rollback

The service is additive; rollback is immediate and has no side effects:

1. Stop the uvicorn process serving port 8011 (or close its terminal).
2. The training queue and all training artifacts are untouched.
3. `dental-caries-detection/ml-service` is unchanged and still serves the old
   RF model; no database, registry, or shared-volume state was modified.
4. If a future integration points the production service at the new path, keep
   the RF path as the default behind a feature flag and revert the flag.

## Limitations (must travel with any reported number)

- Single centre, ~400 cross-validation cases; internal test halves only.
- Panoramic-derived tooth crops; no external cohort yet.
- Probabilities are uncalibrated until all five folds exist and
  `pipeline.py calibrate` is run on the peeked set (diagnostic only).
- Fold 3 of the first run continued from an interrupted checkpoint (documented
  in the main report); its weights are valid but the trajectory is not
  bit-identical to an uninterrupted run.
- This is a technical staging milestone, not evidence of clinical utility.
