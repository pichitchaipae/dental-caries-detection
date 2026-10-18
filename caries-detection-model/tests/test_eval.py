"""Self-tests for the evaluation stage (E1/E1b helpers, E4, E5).

Run directly (pytest is not installed in this environment):
    python tests\\test_eval.py
Every artifact is written to a temporary directory, so the tests never touch
the real outputs of a training run.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import pipeline as P  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t_eval_"))
FAILS = []


def check(name, cond, extra=""):
    print("%-46s %s %s" % (name, "PASS" if cond else "FAIL", extra))
    if not cond:
        FAILS.append(name)


def make_oof(path, n_cases=40, teeth=25, seed=0, skill=0.55, shift=0.0):
    """Synthetic OOF csv: 5 folds, case-grouped, label-correlated scores."""
    rng = np.random.RandomState(seed)
    cases = list(range(1000, 1000 + n_cases))
    fold_of = {c: i % 5 for i, c in enumerate(cases)}
    hdr = (["case_id", "tooth_fdi", "fold"]
           + ["true_%s" % s for s in P.SURFACES]
           + ["prob_%s" % s for s in P.SURFACES]
           + ["pred_%s" % s for s in P.SURFACES])
    rows = [hdr]
    for c in cases:
        for k in range(teeth):
            y = [1 if rng.rand() < p else 0 for p in (0.06, 0.045, 0.026)]
            pr = []
            for j, yy in enumerate(y):
                z = rng.randn() * 1.2 + skill * (2.0 * yy - 1) + shift
                pr.append(float(1.0 / (1.0 + np.exp(-z))))
            d = [1 if pr[j] >= 0.35 else 0 for j in range(3)]
            rows.append([c, 11 + k, fold_of[c]] + y
                        + [round(v, 4) for v in pr] + d)
    p = Path(path)
    p.write_text("\n".join(",".join(str(v) for v in r) for r in rows) + "\n",
                 encoding="utf-8")
    return p


def main():
    # ---- E1 : oof-report -------------------------------------------------
    a = make_oof(TMP / "oofA.csv", seed=1, skill=0.55)
    b = make_oof(TMP / "oofB.csv", seed=1, skill=0.10)      # deliberately worse
    rep_p = TMP / "repA.json"
    P.main(["oof-report", "--oof", str(a), "--n-boot", "60",
            "--out", str(rep_p)])
    rep = json.loads(rep_p.read_text(encoding="utf-8"))
    ok = True
    for s in P.SURFACES:
        r = rep["per_surface"][s]
        ok &= 0.0 <= r["recall_pooled"] <= 1.0
        ok &= 0.0 <= r["precision_pooled"] <= 1.0
        w = r["recall_wilson"]
        ok &= (w[0] <= r["recall_pooled"] <= w[1]) and 0.0 <= w[0] <= w[1] <= 1.0
        ok &= r["ap_fold_mean"] >= 0.0 and r["auroc_fold_mean"] >= 0.5
        ok &= r["prevalence_mean"] > 0.0
    check("E1 oof-report ranges + Wilson brackets", ok)
    check("E1 per-fold AP never pooled into one curve",
          len(rep["per_surface"]["Distal"]["ap_fold_values"]) == 5)
    check("E1 markdown table written beside the json",
          rep_p.with_suffix(".md").exists())

    # ---- E1b : compare-oof ----------------------------------------------
    cmp1 = TMP / "cmp1.json"
    P.main(["compare-oof", "--pair", "%s,%s" % (a, b), "--n-boot", "40",
            "--margin", "0.05", "--seed", "42", "--out", str(cmp1)])
    cmp_ = json.loads(cmp1.read_text(encoding="utf-8"))
    r = cmp_["per_surface"]["Distal"]
    check("E1b better arm shows positive delta-AP", r["delta_ap_mean"] > 0.0,
          "dAP=%+.3f" % r["delta_ap_mean"])
    check("E1b cluster bootstrap excludes zero on a real gap",
          r["delta_ap_boot_ci"][0] > 0.0 or r["delta_ap_boot_ci"][1] < 0.0)
    check("E1b corrected-t interval is finite",
          r["delta_ap_corrected_t_ci"][0] == r["delta_ap_corrected_t_ci"][0])
    check("E1b non-inferiority verdict respects the margin",
          r["non_inferior"] == (r["delta_ap_corrected_t_ci"][0] > -0.05))
    check("E1b k counts fold x seed estimates", r["k_estimates"] == 5)

    cmp2 = TMP / "cmp2.json"
    P.main(["compare-oof", "--pair", "%s,%s" % (a, a), "--n-boot", "20",
            "--margin", "0.05", "--out", str(cmp2)])
    r2 = json.loads(cmp2.read_text(encoding="utf-8"))["per_surface"]["Occlusal"]
    check("E1b self-comparison gives exactly zero delta",
          abs(r2["delta_ap_mean"]) < 1e-12, "dAP=%.3e" % r2["delta_ap_mean"])
    check("E1b identical arms are non-inferior but not superior",
          r2["non_inferior"] and not r2["superior"])

    # mismatched arms must be refused outright
    try:
        c = make_oof(TMP / "oofC.csv", n_cases=30, seed=2, skill=0.4)
        P.main(["compare-oof", "--pair", "%s,%s" % (a, c), "--n-boot", "5",
                "--out", str(TMP / "cmp3.json")])
        check("E1b refuses arms that are not the same teeth", False)
    except SystemExit:
        check("E1b refuses arms that are not the same teeth", True)

    # ---- fix 1 : a wide interval must not be read as equality ------------
    m, lo, hi = P._nb_corrected_t([0.01, -0.02, 0.03, -0.01],
                                  n_train=800, n_test=200)
    check("E1b corrected-t crosses zero for noisy deltas", lo < 0 < hi,
          "[%+.3f,%+.3f]" % (lo, hi))
    m2, lo2, hi2 = P._nb_corrected_t([0.30, 0.28, 0.31, 0.29],
                                     n_train=800, n_test=200)
    check("E1b corrected-t separates a real effect", lo2 > 0,
          "[%+.3f,%+.3f]" % (lo2, hi2))
    _, lo3, _ = P._nb_corrected_t([0.30, 0.28, 0.31, 0.29], n_train=1,
                                  n_test=1)
    check("E1b correction widens with overlapping training sets", lo3 < lo2,
          "lo %.3f -> %.3f" % (lo2, lo3))

    # ---- E4 : kappa ------------------------------------------------------
    krows = ["case_id,tooth_fdi,surface,rater_a,rater_b"]
    rng = np.random.RandomState(7)
    for c in range(1000, 1030):
        for k in range(10):
            for s in P.SURFACES:
                y = 1 if rng.rand() < 0.2 else 0
                f = 0.0 if s != "Occlusal" else 0.4     # Occlusal = noisy raters
                b2 = y if rng.rand() > f else 1 - y
                krows.append("%d,%d,%s,%d,%d" % (c, 11 + k, s, y, b2))
    kp = TMP / "ratings.csv"
    kp.write_text("\n".join(krows) + "\n", encoding="utf-8")
    kap_p = TMP / "kap.json"
    P.main(["kappa", "--pairs", str(kp), "--n-boot", "60", "--out", str(kap_p)])
    kap = json.loads(kap_p.read_text(encoding="utf-8"))
    check("E4 kappa high on clean surfaces",
          kap["per_surface"]["Distal"]["kappa"] > 0.8,
          "%.3f" % kap["per_surface"]["Distal"]["kappa"])
    check("E4 kappa lower on the noisy surface",
          kap["per_surface"]["Occlusal"]["kappa"]
          < kap["per_surface"]["Distal"]["kappa"])
    check("E4 kappa CI brackets the estimate",
          kap["per_surface"]["Distal"]["kappa_boot_ci"][0]
          <= kap["per_surface"]["Distal"]["kappa"]
          <= kap["per_surface"]["Distal"]["kappa_boot_ci"][1])

    krows2 = ["case_id,tooth_fdi,surface,rater_a,rater_b"]
    for c in range(1000, 1010):
        for k in range(4):
            for s in P.SURFACES:
                y = (c + k) % 2
                krows2.append("%d,%d,%s,%d,%d" % (c, 11 + k, s, y, y))
    kp2 = TMP / "ratings_perfect.csv"
    kp2.write_text("\n".join(krows2) + "\n", encoding="utf-8")
    kap2_p = TMP / "kap2.json"
    P.main(["kappa", "--pairs", str(kp2), "--n-boot", "20",
            "--out", str(kap2_p)])
    kap2 = json.loads(kap2_p.read_text(encoding="utf-8"))
    check("E4 perfect agreement gives kappa == 1",
          abs(kap2["per_surface"]["Mesial"]["kappa"] - 1.0) < 1e-9,
          "%.4f" % kap2["per_surface"]["Mesial"]["kappa"])

    # ---- E5 : sample-size -------------------------------------------------
    sz_p = TMP / "sz.json"
    P.main(["sample-size", "--prevalence", "Distal=0.061",
            "--prevalence", "Mesial=0.045", "--prevalence", "Occlusal=0.026",
            "--teeth-per-case", "30", "--design-effect", "1.3",
            "--out", str(sz_p)])
    sz = json.loads(sz_p.read_text(encoding="utf-8"))
    check("E5 rarest surface drives the sample size",
          sz["cases_needed_worst_surface"]
          == sz["per_surface"]["Occlusal"]["cases_needed"])
    check("E5 rarer prevalence needs more cases",
          sz["per_surface"]["Occlusal"]["cases_needed"]
          > sz["per_surface"]["Distal"]["cases_needed"])
    check("E5 design effect inflates the count",
          sz["per_surface"]["Occlusal"]["positives_needed_with_design_effect"]
          > sz["per_surface"]["Occlusal"]["positives_needed"])
    check("E5 plausible external size",
          40 <= sz["cases_needed_worst_surface"] <= 200,
          "%d cases" % sz["cases_needed_worst_surface"])

    # ---- E1 helper : Wilson interval sanity ------------------------------
    lo_w, hi_w = P._wilson_ci(8, 10)
    check("E1 Wilson interval brackets the point estimate", lo_w < 0.8 < hi_w,
          "[%.3f,%.3f]" % (lo_w, hi_w))
    check("E1 Wilson interval handles zero successes",
          P._wilson_ci(0, 10)[0] == 0.0)

    print("\n", "ALL-EVAL-TESTS-PASS" if not FAILS
          else "FAILURES: %s" % FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
