"""Baseline LightGBM submission: raw features, no imputation, native NaN handling.

Deliberately minimal. This is the number every later pipeline component must beat:
5-fold stratified CV AUC 0.9635 (measured twice, independently).

Run:  pixi run python -m smartphone_addiction.baseline
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import polars as pl

from smartphone_addiction.paths import SUBMISSIONS_DIR, TEST_CSV, TRAIN_CSV

CATS = ["gender", "stress_level", "academic_work_impact"]
# Explicit, stable category codes. Polars' physical codes depend on value-encounter
# order, so deriving them per-file would silently misalign train and test.
CAT_CODES: dict[str, dict[str, int]] = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "stress_level": {"Low": 0, "Medium": 1, "High": 2},
    "academic_work_impact": {"No": 0, "Yes": 1},
}

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "verbosity": -1,
    "num_threads": 8,
    "seed": 42,
}
BEST_ROUNDS = 3564  # from lgb.cv with 100-round early stopping; CV AUC 0.963947


def encode(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(c).replace_strict(CAT_CODES[c], default=None, return_dtype=pl.Int32)
            for c in CATS
        ]
    )


def main() -> None:
    t0 = time.time()
    train = encode(pl.read_csv(TRAIN_CSV))
    test = encode(pl.read_csv(TEST_CSV))

    y = train["addicted_label"].to_numpy()
    feats = [c for c in train.columns if c not in ("id", "addicted_label")]
    assert feats == [c for c in test.columns if c != "id"], "train/test column mismatch"

    x_train = train.select(feats).to_numpy().astype(np.float32)
    x_test = test.select(feats).to_numpy().astype(np.float32)
    cat_idx = [feats.index(c) for c in CATS]
    print(
        f"train={x_train.shape} test={x_test.shape} pos_rate={y.mean():.4f}", flush=True
    )

    ds = lgb.Dataset(x_train, y, categorical_feature=cat_idx, free_raw_data=False)
    model = lgb.train(PARAMS, ds, num_boost_round=BEST_ROUNDS)
    # predict() is typed as a union including sparse types; this path is always dense.
    pred = np.asarray(model.predict(x_test), dtype=np.float64)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / "lgbm_baseline.csv"
    pl.DataFrame({"id": test["id"], "addicted_label": pred}).write_csv(out)

    print(
        f"wrote {out}  rows={len(pred):,}  "
        f"pred[min={np.min(pred):.4f} mean={np.mean(pred):.4f} "
        f"max={np.max(pred):.4f}]  "
        f"[{time.time() - t0:.0f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
