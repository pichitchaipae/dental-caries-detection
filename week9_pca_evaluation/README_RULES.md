# README — Rules & Constraints

## ⚠️ STRICT SURFACE CLASSIFICATION RULE

> **"Surface Incorrect" classes MUST be strictly limited to:**
>
> ```
> ['Distal', 'Mesial', 'Occlusal']
> ```
>
> **No other classes may be generated, predicted, or logged.**

### Enforcement

- `normalize_surface()` and `normalize_surface_fine()` in
  `evaluation_engine.py` route all labels through `enforce_surface_rule()`.
- Any label not in `ALLOWED_SURFACE_CLASSES = ["Distal", "Mesial", "Occlusal"]`
  is remapped to `"Unclassified"` and a warning is printed.
- After filtering, an **assertion** verifies that no unexpected classes
  remain in the 3-class evaluation pool.

### Why This Matters

The evaluation metrics (Accuracy, Precision, Recall, F1) are computed
on exactly 3 classes.  Introducing additional classes (e.g. "Buccal",
"Lingual", "Cervical") would silently corrupt the confusion matrix and
all derived statistics.

### PCA Methods Evaluated (Week 9)

| Method | Name                    | Description                                      |
|--------|-------------------------|--------------------------------------------------|
| 0      | `baseline_opencv`       | Original OpenCV PCA — no heuristics, no clamping |
| 1      | `square_heuristic`      | Eigenvalue-ratio check for square teeth          |
| 2      | `max_span`              | Maximum projected span on each eigenvector       |
| 3      | `split_centroid`        | Anatomical upper/lower centroid vector            |
| 5      | `vertical_prior`        | Absolute vertical prior + 3-rule logic           |

Method 4 is a **placeholder** and is excluded from evaluation.

### Team Checklist

- [ ] All new surface labels must be reviewed against the 3-class rule.
- [ ] Any pipeline change that adds a surface class must update
      `ALLOWED_SURFACE_CLASSES` in `evaluation_engine.py` **and** this
      README.
- [ ] Run `python run_evaluation.py --sample 311,33` to verify no
      assertion errors before committing.

---
*Last updated: 2026-02-23*
