"""A4 self-test: per-epoch durability, completed-fold reuse, mid-fold resume.

CPU-only smoke test on a tiny slice of the real cache (no Graphical Processing
Unit). All artifacts are written to a temporary directory; the real outputs/
of a training run are never touched.

Run from caries-detection-model/:
    D:\\Users\\jaopi\\anaconda3\\envs\\sp_project\\python.exe tests\\test_resume.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import pipeline as P  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t_resume_"))
FAILS = []


def check(name, cond, extra=""):
    print("%-52s %s %s" % (name, "PASS" if cond else "FAIL", extra))
    if not cond:
        FAILS.append(name)


def tiny_fold(n_train_cases=3, n_val_cases=4):
    _, samples = P.load_manifest()
    by_case = {}
    for s in samples:
        by_case.setdefault(int(s["case_id"]), []).append(s)
    pos = {c: sum(1 for s in by_case[c] if any(s["label"])) for c in by_case}
    ranked = sorted(by_case, key=lambda c: (-pos[c], c))
    val_c = ranked[:n_val_cases]
    tr_c = [c for c in sorted(by_case) if c not in val_c][:n_train_cases]
    tr = [s for c in tr_c for s in by_case[c]]
    va = [s for c in val_c for s in by_case[c]]
    return tr, va


def make_cfg(epochs=2, recall_target=0.80):
    return P.TrainConfig(backbone="efficientnet_b0", ablation="pca", loss="focal",
                         use_metadata=False, epochs=epochs, batch_size=16,
                         lr=3e-4, seed=42, cv="single", num_workers=0,
                         patience=None, recall_target=recall_target)


def main():
    import torch

    torch.cuda.is_available = lambda: False          # force CPU: smoke only
    P.OUTPUTS_DIR = TMP
    (TMP / "outputs-placeholder").write_text("x", encoding="utf-8")

    tr, va = tiny_fold()
    n_pos = sum(1 for s in tr + va if any(s["label"]))
    print("tiny slice: %d train teeth, %d val teeth, %d positive teeth"
          % (len(tr), len(va), n_pos))
    pre = P.ablation_to_pre("pca", img_size=160)

    # ---- clean full run (2 epochs) --------------------------------
    r_clean = P.train_fold(tr, va, make_cfg(), pre, out_prefix="clean")
    check("clean run wrote checkpoint", (TMP / "clean.pt").is_file())
    check("clean run wrote last.pt", (TMP / "clean_last.pt").is_file())
    check("clean run wrote metrics.json", (TMP / "clean_metrics.json").is_file())
    check("clean run wrote history.json", (TMP / "clean_history.json").is_file())

    # ---- completed-fold reuse (checkpoint + metrics exist) ---------
    r_done = P.train_fold(tr, va, make_cfg(), pre, out_prefix="clean",
                          resume=True)
    check("reuse: test probabilities identical",
          np.allclose(r_clean["test_y_prob"], r_done["test_y_prob"]))
    check("reuse: thresholds identical",
          np.allclose(r_clean["thresholds"], r_done["thresholds"]))
    check("reuse: test rows identical",
          r_clean["test_keys"] == r_done["test_keys"])
    check("reuse: macroF1 identical",
          abs(r_clean["macro_f1"] - r_done["macro_f1"]) < 1e-9)

    # ---- mid-fold resume after a simulated power cut ---------------
    orig = P._save_fold_state
    state = {"n": 0}

    def killer(*a, **k):
        orig(*a, **k)
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt("simulated power cut")

    P._save_fold_state = killer
    try:
        P.train_fold(tr, va, make_cfg(), pre, out_prefix="cut")
    except KeyboardInterrupt:
        pass
    finally:
        P._save_fold_state = orig
    check("interrupted run left last.pt", (TMP / "cut_last.pt").is_file())
    check("interrupted run has no final checkpoint",
          not (TMP / "cut.pt").is_file())
    check("interrupted run kept epoch history",
          (TMP / "cut_history.json").is_file())

    r_res = P.train_fold(tr, va, make_cfg(), pre, out_prefix="cut", resume=True)
    check("resumed run wrote final checkpoint", (TMP / "cut.pt").is_file())
    d = float(np.max(np.abs(r_res["test_y_prob"] - r_clean["test_y_prob"])))
    check("resumed == uninterrupted (max prob diff < 1e-4)", d < 1e-4,
          "diff=%.2e" % d)
    check("resumed best_epoch == uninterrupted",
          r_res["best_epoch"] == r_clean["best_epoch"],
          "%s vs %s" % (r_res["best_epoch"], r_clean["best_epoch"]))
    check("resumed thresholds nested",
          P.verify_nested_thresholds([r_res], make_cfg().recall_target) is True)

    # ---- a tampered checkpoint must be refused ----------------------
    # (note: on a tiny slice, changing --recall-target alone can legitimately
    # reproduce the same threshold vector, so the guard is tested by tampering
    # the stored thresholds directly)
    ck = torch.load(TMP / "clean.pt", map_location="cpu")
    ck["thresholds"] = [0.99, 0.99, 0.99]
    torch.save(ck, TMP / "tampered.pt")
    shutil.copyfile(TMP / "clean_metrics.json", TMP / "tampered_metrics.json")
    shutil.copyfile(TMP / "clean_thresholds.json", TMP / "tampered_thresholds.json")
    refused = False
    try:
        P.train_fold(tr, va, make_cfg(), pre, out_prefix="tampered",
                     resume=True)
    except SystemExit:
        refused = True
    check("tampered thresholds refused", refused)

    print("\ntemp dir:", TMP)
    if FAILS:
        print("FAILED:", FAILS)
        raise SystemExit(1)
    print("ALL-RESUME-TESTS-PASS")


if __name__ == "__main__":
    main()
