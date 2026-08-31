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
slice, never on the rows being scored.

Run:  pixi run python -m smartphone_addiction.pipeline
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from smartphone_addiction.paths import (
    PROCESSED_DIR,
    SUBMISSIONS_DIR,
    TEST_CSV,
    TRAIN_CSV,
)

SEED = 42
N_FOLDS = 5
SMOOTHING = 30.0  # target-encoding prior weight
MISSING_KEY = -999.0  # NaN gets its own encoding level rather than being dropped

CATS = ["gender", "stress_level", "academic_work_impact"]
CAT_CODES: dict[str, dict[str, int]] = {
    "gender": {"Female": 0, "Male": 1, "Other": 2},
    "stress_level": {"Low": 0, "Medium": 1, "High": 2},
    "academic_work_impact": {"No": 0, "Yes": 1},
}
FRACTIONAL = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time",
]
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_bin": 2047,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "verbosity": -1,
    "num_threads": 8,
    "seed": SEED,
}


def _encode_cats(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(c).replace_strict(CAT_CODES[c], default=None, return_dtype=pl.Int32)
            for c in CATS
        ]
    )


def _key(col: np.ndarray) -> np.ndarray:
    """Exact-value key with NaN as its own level. Rounded float, never str."""
    return np.where(np.isnan(col), MISSING_KEY, np.round(col.astype(np.float64), 6))


def _te_fit(keys: np.ndarray, y: np.ndarray) -> tuple[dict[float, float], float]:
    prior = float(y.mean())
    agg = (
        pl.DataFrame({"k": keys, "y": y.astype(np.float64)})
        .group_by("k")
        .agg(pl.col("y").mean().alias("m"), pl.len().alias("n"))
        .with_columns(
            (
                (pl.col("m") * pl.col("n") + prior * SMOOTHING)
                / (pl.col("n") + SMOOTHING)
            ).alias("te")
        )
    )
    return dict(zip(agg["k"].to_list(), agg["te"].to_list(), strict=True)), prior


def _te_apply(lut: dict[float, float], prior: float, keys: np.ndarray) -> np.ndarray:
    return np.array([lut.get(v, prior) for v in keys.tolist()], dtype=np.float32)


def _structural(a: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    """Constraint residuals, missingness-conditional bounds, and the decimal lattice."""
    daily = a[:, idx["daily_screen_time_hours"]]
    comp = a[
        :, [idx["social_media_hours"], idx["gaming_hours"], idx["work_study_hours"]]
    ]
    parts0 = np.nansum(comp, axis=1)  # NaN treated as 0: always defined
    parts_n = comp.sum(axis=1)  # NaN-propagating: defined only when complete
    n_missing = np.isnan(comp).sum(axis=1)
    other = daily - parts0
    cols = [
        other,
        daily - parts_n,
        other / np.clip(daily, 0.1, None),
        parts0 / np.clip(daily, 0.1, None),
        a[:, idx["weekend_screen_time"]] - other,
        # Bounds live only inside their own missingness cell, NaN elsewhere, so LightGBM
        # routes each case down its own learned default direction.
        np.where(np.isnan(daily) & (n_missing == 0), parts0, np.nan),
        np.where(~np.isnan(daily) & (n_missing == 1), daily - parts0, np.nan),
        np.where(~np.isnan(daily) & (n_missing == 0), daily - parts0, np.nan),
    ]
    for c in FRACTIONAL:
        v = a[:, idx[c]]
        cols.append(np.round(v - np.floor(v), 2))
        cols.append(np.round(v * 10) % 10)
    return np.column_stack(cols).astype(np.float32)


def main() -> None:
    t0 = time.time()
    train = _encode_cats(pl.read_csv(TRAIN_CSV))
    test = _encode_cats(pl.read_csv(TEST_CSV))
    feats = [c for c in train.columns if c not in ("id", "addicted_label")]
    idx = {c: i for i, c in enumerate(feats)}
    cat_idx = [idx[c] for c in CATS]

    x = train.select(feats).to_numpy().astype(np.float32)
    x_test = test.select(feats).to_numpy().astype(np.float32)
    y = train["addicted_label"].to_numpy()
    n_num = len([c for c in feats if c not in CATS])

    # Frequency encoding: no target involved, so pooling train+test is transductive.
    freq = np.zeros((len(y), len(feats)), dtype=np.float32)
    freq_test = np.zeros((len(x_test), len(feats)), dtype=np.float32)
    for j in range(len(feats)):
        k_tr, k_te = _key(x[:, j]), _key(x_test[:, j])
        vals, counts = np.unique(np.concatenate([k_tr, k_te]), return_counts=True)
        lut = dict(zip(vals.tolist(), counts.tolist(), strict=True))
        freq[:, j] = [lut.get(v, 0) for v in k_tr.tolist()]
        freq_test[:, j] = [lut.get(v, 0) for v in k_te.tolist()]

    keys = np.column_stack([_key(x[:, j]) for j in range(n_num)])
    keys_test = np.column_stack([_key(x_test[:, j]) for j in range(n_num)])
    struct, struct_test = _structural(x, idx), _structural(x_test, idx)

    oof = np.zeros(len(y))
    test_ranks = np.zeros(len(x_test))
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(x, y)
    for k, (tr, va) in enumerate(folds):
        te_tr = np.zeros((len(tr), n_num), dtype=np.float32)
        te_va = np.zeros((len(va), n_num), dtype=np.float32)
        te_test = np.zeros((len(keys_test), n_num), dtype=np.float32)
        inner = StratifiedKFold(5, shuffle=True, random_state=0)
        for j in range(n_num):
            k_all, y_all = keys[tr, j], y[tr]
            for i_tr, i_va in inner.split(k_all, y_all):
                lut, prior = _te_fit(k_all[i_tr], y_all[i_tr])
                te_tr[i_va, j] = _te_apply(lut, prior, k_all[i_va])
            lut, prior = _te_fit(k_all, y_all)
            te_va[:, j] = _te_apply(lut, prior, keys[va, j])
            te_test[:, j] = _te_apply(lut, prior, keys_test[:, j])

        x_tr = np.hstack([x[tr], freq[tr], te_tr, struct[tr]]).astype(np.float32)
        x_va = np.hstack([x[va], freq[va], te_va, struct[va]]).astype(np.float32)
        x_ts = np.hstack([x_test, freq_test, te_test, struct_test]).astype(np.float32)

        fit_i, stop_i = train_test_split(
            np.arange(len(tr)), test_size=0.1, random_state=SEED + k, stratify=y[tr]
        )
        ds = lgb.Dataset(
            x_tr[fit_i], y[tr][fit_i], categorical_feature=cat_idx, free_raw_data=False
        )
        dv = lgb.Dataset(
            x_tr[stop_i],
            y[tr][stop_i],
            categorical_feature=cat_idx,
            reference=ds,
            free_raw_data=False,
        )
        model = lgb.train(
            PARAMS,
            ds,
            num_boost_round=12000,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = np.asarray(model.predict(x_va))
        # Rank-average across folds: fold models are differently calibrated and AUC only
        # reads order, so averaging ranks is sounder than averaging probabilities.
        test_ranks += rankdata(np.asarray(model.predict(x_ts))) / (N_FOLDS * len(x_ts))
        print(
            f"  fold {k} rounds={model.best_iteration} "
            f"auc={roc_auc_score(y[va], oof[va]):.6f}",
            flush=True,
        )

    auc = roc_auc_score(y, oof)
    print(f"\nOOF AUC = {auc:.6f}  [{time.time() - t0:.0f}s]", flush=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "oof_pipeline.npy", oof)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / "lgbm_pipeline.csv"
    pl.DataFrame({"id": test["id"], "addicted_label": test_ranks}).write_csv(out)
    print(f"wrote {out}  rows={len(test_ranks):,}", flush=True)


if __name__ == "__main__":
    main()
