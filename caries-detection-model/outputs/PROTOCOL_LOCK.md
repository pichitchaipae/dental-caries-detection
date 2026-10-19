# Protocol lock — reruns, transfer matrix, Stage-2 repeats (locked 16 Oct 2026)

Primary metric for ADOPTING a new arm or config: paired bootstrap 95%
Confidence Interval of delta Average Precision (new minus reference) on
IDENTICAL test teeth (`compare` command, same seed and split); adopt if and
only if the lower bound exceeds zero. Report per-surface deltas regardless.
Recall at tuned thresholds must not regress — reported, not a gate.

Secondary and reporting: test-half Recall, Precision, F1-score plus
case-level bootstrap Confidence Interval; Cross-Validation mean and standard
deviation where folds exist; out-of-fold pooled decisions with per-fold
Average Precision mean and standard deviation (never pooled probabilities).

Splits and seeds, frozen: seed 42 single split plus nested tune and test
halves (seed plus 7919); repeated stratified group k-fold (3 repeats of 5
folds, seeds 42, 1042, 2042); 100 peeked cases excluded from ALL training
and evaluation runs (`--exclude`).

Thresholds: tuned per fold on tune data only (runtime-enforced by
`verify_nested_thresholds`), default target Recall 0.80 (margin 0.85 under
test); never from test data.

Epoch selection: tune macro Average Precision, threshold-free; early
stopping patience 8; plateau scheduler on tune loss.

Out-of-fold: decisions pooled across folds with per-fold thresholds;
Average Precision per fold, never pooled probabilities.

Labels: Extensible Markup Language reference as is; no relabeling from
model outputs (blinded expert review only).

Change log: thresholds grid 0.05 to 0.95 (unchanged); bootstrap 1000
resamples at case level (unchanged).

---

# Evaluation stage lock v2 (locked 17 Oct 2026, before any arm decision)

Decision rule between two arms (replaces "interval crosses zero"): an arm is
adopted when the LOWER bound of the Nadeau & Bengio corrected t interval of
delta Average Precision exceeds minus the declared margin. Margin = 0.05,
locked here, before the runs. An interval that crosses zero is "not yet
separated", never "equal". Distinguishability is additionally reported from a
patient-level cluster bootstrap; it is not the adoption gate.

Training variance: one seed cannot compare training methods. The pair
`pca` against `off` is run on seeds 42, 43 (and 44 if the schedule allows);
k = number of folds times number of seeds in the corrected t test. If the
replicates do not complete, the report must state that the comparison excludes
training variance.

Metrics reported per surface: prevalence (reference Average Precision of
random guessing), Average Precision per fold as mean and standard deviation
with a cluster bootstrap interval, lift (Average Precision over prevalence),
AUROC, and pooled Recall/Precision/F1 from nested per-fold decisions with both
a Wilson score interval and a patient-level cluster bootstrap interval.
Pooled probabilities are still never concatenated into one curve.

Thresholds for the delivered ensemble (`calibrate`): temperature scaling per
surface fitted on the 100 peeked cases, which every fold excluded from
training, then the threshold on the calibrated ensemble probability. Tuning
target for delivery is Recall 0.85 because a threshold chosen as the largest
one still meeting 0.80 on one sample regresses below 0.80 on the next.
Peeked-set numbers are diagnostic only and must never be quoted as performance.

Calibration reporting: Brier score before and after temperature scaling plus a
10-bin reliability diagram, reported on the external set once it exists.

Label ceiling: inter-rater Cohen kappa per surface on 30 to 50 cases read
blindly by two dentists (`kappa`). A low kappa on a surface means that
surface's Average Precision is limited by the reference standard.

External test: size from the measured prevalence with design effect 1.3
(`sample-size`, expected about 103 cases driven by Occlusal), used exactly
once, reporting sensitivity, precision, intervals, calibration and subgroups.
End-to-end (go or no-go): report both crop level (matched teeth) and system
level (unmatched teeth scored as probability zero; quadrant errors counted by
comparing left/right sides on the image), and apply the same margin rule to
delta Average Precision between human-annotated masks and predicted masks.
Detection caveat to disclose: it is unknown whether the Detectron2 weights ever
saw these 500 cases.

Reporting checklists: CLAIM and TRIPOD+AI.
