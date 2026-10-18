"""Unit tests: measurement-protocol invariants (no Graphics Processing Unit training).

Run from caries-detection-model/:
  D:\\Users\\jaopi\\anaconda3\\envs\\sp_project\\python.exe tests/test_protocol.py
(no pytest dependency: plain asserts + tiny runner.)
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _syn_fold(seed, n=120):
    rng = np.random.RandomState(seed)
    yt = (rng.rand(n, 3) < [0.3, 0.2, 0.1]).astype(int)
    yp = np.clip(yt * 0.5 + rng.rand(n, 3) * 0.4, 0, 1)
    return yt, yp


def test_thresholds_from_inner_data():
    from pipeline import tune_thresholds, verify_nested_thresholds

    results = []
    for fi in range(2):
        yt, yp = _syn_fold(fi)
        results.append({"tune_y_true": yt, "tune_y_prob": yp,
                        "thresholds": tune_thresholds(yt, yp, 0.80)})
    assert verify_nested_thresholds(results, 0.80) is True


def test_thresholds_reject_test_tuning():
    from pipeline import tune_thresholds, verify_nested_thresholds

    yt, yp = _syn_fold(1)
    good = tune_thresholds(yt, yp, 0.80)
    bad = list(good)
    bad[0] = min(1.0, bad[0] + 0.05)  # tampered: cannot come from tuning
    try:
        verify_nested_thresholds([{"tune_y_true": yt, "tune_y_prob": yp,
                                   "thresholds": bad}], 0.80)
    except AssertionError:
        pass
    else:
        raise SystemExit("FAIL: tampered thresholds accepted")


def test_group_partition_no_leak():
    from pipeline import _split_indices

    rng = np.random.RandomState(2)
    cases = np.repeat(np.arange(50), 4)  # 4 teeth per patient
    strata = (rng.rand(200) < 0.2).astype(int).tolist()
    for tr, va in _split_indices(cases, 9, "stratified-groupkfold",
                                 strata=strata):
        assert not (set(cases[tr]) & set(cases[va]))
    assert len(_split_indices(cases, 9, "repeated-stratified-groupkfold",
                             strata=strata, repeats=2)) == 10


def test_threshold_determinism():
    from pipeline import tune_thresholds

    yt, yp = _syn_fold(3)
    assert tune_thresholds(yt, yp, 0.85) == tune_thresholds(yt, yp, 0.85)


if __name__ == "__main__":
    test_thresholds_from_inner_data()
    test_thresholds_reject_test_tuning()
    test_group_partition_no_leak()
    test_threshold_determinism()
    print("ALL-PROTOCOL-TESTS-PASS")
