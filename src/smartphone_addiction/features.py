"""Shared representation layer: binning keys, value-level encoding, generator structure.

Extracted verbatim from `pipeline.py` so that `members.py` can build feature-SUBSET
variants on the same code path. The subsets are the point: the experiment ledger found
that removing a whole information channel produces a genuinely different solution and
earns stack weight, while hyperparameter variants earn nothing. That comparison is only
honest if every arm shares one implementation of the features it does keep.

THE THREE CHANNELS, and what each buys (measured, see README):
  raw     the 12 competition columns. NaN stays native, so LightGBM learns a
          per-split default direction rather than trusting a point estimate.
  freq    value-frequency counts. No target involved, so pooling train+test
          is transductive, not leaky.
  te      value-level target encoding. +0.00122. Touches the label, so it is
          fold-nested twice.
  struct  generator-constraint residuals and the decimal lattice. +0.00105.

LEAKAGE. `fold_target_encoding` is the only function here that sees `y`. An outer fold's
validation rows are encoded from that fold's training rows alone, and those
training rows
are encoded through an inner 5-fold so no row contributes to its own encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

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


def encode_cats(df: pl.DataFrame) -> pl.DataFrame:
    """Explicit, stable category codes.

    Polars' physical codes depend on value-encounter order, so deriving them per-file
    would silently misalign train and test.
    """
    return df.with_columns(
        [
            pl.col(c).replace_strict(CAT_CODES[c], default=None, return_dtype=pl.Int32)
            for c in CATS
        ]
    )


def key(col: np.ndarray) -> np.ndarray:
    """Exact-value key with NaN as its own level. Rounded float, never str."""
    return np.where(np.isnan(col), MISSING_KEY, np.round(col.astype(np.float64), 6))


def te_fit(keys: np.ndarray, y: np.ndarray) -> tuple[dict[float, float], float]:
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


def te_apply(lut: dict[float, float], prior: float, keys: np.ndarray) -> np.ndarray:
    return np.array([lut.get(v, prior) for v in keys.tolist()], dtype=np.float32)


def structural(a: np.ndarray, idx: dict[str, int]) -> np.ndarray:
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


@dataclass
class Static:
    """Everything computable without seeing a fold's labels."""

    feats: list[str]
    idx: dict[str, int]
    cat_idx: list[int]
    n_num: int
    x: np.ndarray
    x_test: np.ndarray
    y: np.ndarray
    test_ids: pl.Series
    freq: np.ndarray
    freq_test: np.ndarray
    keys: np.ndarray
    keys_test: np.ndarray
    struct: np.ndarray
    struct_test: np.ndarray


def prepare(train: pl.DataFrame, test: pl.DataFrame) -> Static:
    """Build every fold-independent block once, so all member arms share it."""
    train, test = encode_cats(train), encode_cats(test)
    feats = [c for c in train.columns if c not in ("id", "addicted_label")]
    idx = {c: i for i, c in enumerate(feats)}
    cat_idx = [idx[c] for c in CATS]
    numeric = [c for c in feats if c not in CATS]
    num_idx = [idx[c] for c in numeric]
    n_num = len(numeric)

    x = train.select(feats).to_numpy().astype(np.float32)
    x_test = test.select(feats).to_numpy().astype(np.float32)
    y = train["addicted_label"].to_numpy()

    # Frequency encoding: no target involved, so pooling train+test is transductive.
    freq = np.zeros((len(y), len(feats)), dtype=np.float32)
    freq_test = np.zeros((len(x_test), len(feats)), dtype=np.float32)
    for j in range(len(feats)):
        k_tr, k_te = key(x[:, j]), key(x_test[:, j])
        vals, counts = np.unique(np.concatenate([k_tr, k_te]), return_counts=True)
        lut = dict(zip(vals.tolist(), counts.tolist(), strict=True))
        freq[:, j] = [lut.get(v, 0) for v in k_tr.tolist()]
        freq_test[:, j] = [lut.get(v, 0) for v in k_te.tolist()]

    # Value-encoding keys cover the numeric columns, selected by NAME. The original
    # pipeline sliced the first n_num columns positionally, which is only correct while
    # the CSV happens to place every numeric column before every categorical one.
    keys = np.column_stack([key(x[:, j]) for j in num_idx])
    keys_test = np.column_stack([key(x_test[:, j]) for j in num_idx])
    return Static(
        feats=feats,
        idx=idx,
        cat_idx=cat_idx,
        n_num=n_num,
        x=x,
        x_test=x_test,
        y=y,
        test_ids=test["id"],
        freq=freq,
        freq_test=freq_test,
        keys=keys,
        keys_test=keys_test,
        struct=structural(x, idx),
        struct_test=structural(x_test, idx),
    )


def folds(y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED):
    """The one fold contract every arm must share, so comparisons stay paired."""
    return StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y
    )


def fold_target_encoding(
    s: Static, tr: np.ndarray, va: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Doubly-nested target encoding for one fold. The only function that sees y."""
    te_tr = np.zeros((len(tr), s.n_num), dtype=np.float32)
    te_va = np.zeros((len(va), s.n_num), dtype=np.float32)
    te_test = np.zeros((len(s.keys_test), s.n_num), dtype=np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=0)
    for j in range(s.n_num):
        k_all, y_all = s.keys[tr, j], s.y[tr]
        for i_tr, i_va in inner.split(k_all, y_all):
            lut, prior = te_fit(k_all[i_tr], y_all[i_tr])
            te_tr[i_va, j] = te_apply(lut, prior, k_all[i_va])
        lut, prior = te_fit(k_all, y_all)
        te_va[:, j] = te_apply(lut, prior, s.keys[va, j])
        te_test[:, j] = te_apply(lut, prior, s.keys_test[:, j])
    return te_tr, te_va, te_test
