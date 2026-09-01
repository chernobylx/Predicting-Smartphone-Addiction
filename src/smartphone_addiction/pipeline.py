"""Full pipeline: fine binning, value-level encoding, and generator-structure features.

Measured 5-fold OOF AUC 0.968069, against 0.963659 for the raw-feature baseline.
The gain decomposes into three mechanisms, each verified separately:

  +0.00215  max_bin 2047. Six of nine numeric columns have more distinct values than
            LightGBM's default 255 bins (weekend_screen_time 1460, daily 1398), and
            the label rule has hard steps at exact 2-decimal values. At 255 bins the
            bucket straddling daily=8.00 holds ~2,300 rows it cannot separate.
  +0.00122  Value-level encoding. notifications_per_day has raw univariate AUC
            0.5079 but 0.7486 target-encoded: adjacent integer values differ in
            target rate by 0.22 against a sampling sd of 0.008. These columns are
            lookup keys, not quantities.
  +0.00105  Generator structure. daily >= social+gaming+work holds for 100.000% of
            complete rows, so the residual is a real hidden quantity and, where a
            term is missing, the constraint bounds it. Plus the first decimal
            digit, a generator sampling fingerprint.

LEAKAGE DISCIPLINE. Target encoding is the one component that touches the label, and
it is nested twice: an outer fold's validation rows are encoded only from that fold's
training rows, and those training rows are themselves encoded through an inner 5-fold
so no row contributes to its own encoding. Frequency encoding uses no target at all, so
pooling train+test is transductive, not leaky. Early stopping runs on a per-fold inner
slice, never on the rows being scored. `tests/test_members.py` holds that claim to a
shuffled-label test: under a random target this pipeline must score chance.

This module is the `full` arm of `members.py` — same folds, same features, one
implementation — run on its own and written out as a standalone submission.

Run:  pixi run python -m smartphone_addiction.pipeline
"""

from __future__ import annotations

import numpy as np
import polars as pl

from smartphone_addiction.features import prepare
from smartphone_addiction.members import ARMS, run_arm, save_member
from smartphone_addiction.paths import (
    PROCESSED_DIR,
    SUBMISSIONS_DIR,
    TEST_CSV,
    TRAIN_CSV,
)

ARM = "full"


def main() -> None:
    s = prepare(pl.read_csv(TRAIN_CSV), pl.read_csv(TEST_CSV))
    print(f"train={s.x.shape} test={s.x_test.shape} pos_rate={s.y.mean():.4f}\n")
    oof, test_ranks, auc = run_arm(s, ARM, ARMS[ARM])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "oof_pipeline.npy", oof)
    # Also register it as a stack member, so `stack.py` can be run straight after.
    save_member(ARM, oof, test_ranks)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / "lgbm_pipeline.csv"
    pl.DataFrame({"id": s.test_ids, "addicted_label": test_ranks}).write_csv(out)
    print(f"wrote {out}  rows={len(test_ranks):,}  OOF AUC={auc:.6f}", flush=True)


if __name__ == "__main__":
    main()
