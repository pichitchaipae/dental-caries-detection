# Backend result fixtures

The v2 fixtures use saved ML inference for case 10, recorded on 2026-10-29:
29 teeth, two predicted lesions (FDI 45 and 26, both distal). These are model
predictions, not annotation masks or a claim of diagnostic correctness.

Sources, relative to the application directory:
- `outputs/notebook-final-review-20261029/case10_native_service.json`
  SHA-256: `5189b11307a9d7754ce5c92db6bfdce0eccdf6e78f54d399f28724c1b89f0441`
- `outputs/notebook-final-review-20261029/case10_native_research.json`
  SHA-256: `fa5dbc7fe03259c09133b5299486cfeebba03af68927f3c7b80f57cb0fd24f7e`

The raw fixture preserves the full service result geometry, tooth ordering,
metadata, lesion confidence and centroids, without rounding coordinates.
Each lesion was matched to the research result by tooth order/FDI, tooth
polygon and exact equality of lesion bbox, mask, confidence and centroid.
The research result supplies the recorded per-lesion surface. IDs are assigned
as zero-based indices within each tooth to model the pending ML-10 output;
they were not present in the historical service result. Surface probability
is updated from the historical null value to the maximum lesion confidence
for that surface, as required by the v2 draft. No inference was rerun.

`result.sample.json` data is generated using `adaptInferenceResult()` on the
raw fixture. Its existing `image_base64` is a 1x1 placeholder, **not** the case
10 radiograph; do not use it to validate visual alignment. All polygons use
the original 3037x1536 panorama coordinates. No source radiograph is included.
The source output files may be ignored/unavailable in fresh clones; the
fixtures are self-contained and tests do not require those source files.
