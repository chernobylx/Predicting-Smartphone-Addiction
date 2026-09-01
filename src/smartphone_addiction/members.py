"""Generate the stack's member vectors: one OOF and one test vector per arm.

WHY THIS MODULE EXISTS. `stack.py` reads members from `data/processed/members/`, and
until now no committed code wrote that directory — the 26-member stack behind the
0.97013 leaderboard score was assembled from arms that lived outside the repository.
A reader who cloned this project and followed the README got an empty member set.

WHAT IS AND IS NOT RECOVERABLE. The historical run used 29 candidate arms; the ledger
names only some of them, and one of the weight-earning members was an embedding neural
network whose framework is not even a project dependency. Those recipes are gone.
So this
module does not reconstruct 0.97013. It rebuilds a documented, reduced member set that
reproduces the *finding* — that diversity has to come from what the model sees, not how
it fits — and it states its own honest number.

THE ARMS. Every arm shares one fold contract (`features.folds`), so every comparison
between them is paired on identical rows.

  full        raw + freq + te + struct. The representation pipeline, and the
              strongest single arm.
  no_te       drops target encoding. Earned 0.033 weight historically.
  struct_imp  drops both value-encoding channels, keeping generator structure.
              Earned 0.100.
  raw         the 12 competition columns alone. The baseline representation, same folds.
  xgb_full    XGBoost on the full matrix. Different model CLASS, the other
              thing that paid.
  cat_full    CatBoost on the full matrix. Ordered boosting, a genuinely
              different scheme.

DELIBERATELY OMITTED: `linear_tree`, `extra_trees`, DART and ExtraTrees. All four were
measured and all four earned exactly zero weight — same matrix, same solution, no
independent error to cancel. They are absent because they were refuted, not overlooked.

Run:  pixi run python -m smartphone_addiction.members [--arms full,no_te,...]
      Expect hours on the full 691k rows; each arm is a 5-fold fit.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from smartphone_addiction.features import (
    N_FOLDS,
    SEED,
    Static,
    fold_target_encoding,
    folds,
    prepare,
)
from smartphone_addiction.paths import PROCESSED_DIR, TEST_CSV, TRAIN_CSV

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_bin": 2047,  # the single largest measured win: +0.00215 over the default 255
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "verbosity": -1,
    "num_threads": 8,
    "seed": SEED,
}
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "min_child_weight": 20,
    "max_bin": 2047,
    "tree_method": "hist",
    "nthread": 8,
    "seed": SEED,
}
CAT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.05,
    "depth": 8,
    "border_count": 254,  # CatBoost's ceiling on CPU
    "random_seed": SEED,
    "thread_count": 8,
    "verbose": False,
}
MAX_ROUNDS = 12000
STOPPING = 200


@dataclass(frozen=True)
class Arm:
    blocks: tuple[str, ...]
    model: str


ARMS: dict[str, Arm] = {
    "full": Arm(("raw", "freq", "te", "struct"), "lgb"),
    "no_te": Arm(("raw", "freq", "struct"), "lgb"),
    "struct_imp": Arm(("raw", "struct"), "lgb"),
    "raw": Arm(("raw",), "lgb"),
    "xgb_full": Arm(("raw", "freq", "te", "struct"), "xgb"),
    "cat_full": Arm(("raw", "freq", "te", "struct"), "cat"),
}


def _assemble(
    s: Static, blocks: tuple[str, ...], rows: np.ndarray | None, te: np.ndarray | None
) -> np.ndarray:
    """Concatenate the requested channels. `rows=None` means the test matrix."""
    parts: list[np.ndarray] = []
    for b in blocks:
        if b == "raw":
            parts.append(s.x_test if rows is None else s.x[rows])
        elif b == "freq":
            parts.append(s.freq_test if rows is None else s.freq[rows])
        elif b == "struct":
            parts.append(s.struct_test if rows is None else s.struct[rows])
        elif b == "te":
            assert te is not None, "te block requested but no encoding supplied"
            parts.append(te)
        else:  # pragma: no cover - guarded by ARMS
            raise ValueError(f"unknown block {b!r}")
    return np.hstack(parts).astype(np.float32)


def _fit_predict(
    model: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_stop: np.ndarray,
    y_stop: np.ndarray,
    x_va: np.ndarray,
    x_ts: np.ndarray,
    cat_idx: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit one fold, stopping on a held-aside slice, never on the scored rows."""
    if model == "lgb":
        import lightgbm as lgb

        ds = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
        dv = lgb.Dataset(
            x_stop,
            y_stop,
            categorical_feature=cat_idx,
            reference=ds,
            free_raw_data=False,
        )
        m = lgb.train(
            LGB_PARAMS,
            ds,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dv],
            callbacks=[lgb.early_stopping(STOPPING, verbose=False)],
        )
        return (
            np.asarray(m.predict(x_va)),
            np.asarray(m.predict(x_ts)),
            int(m.best_iteration),
        )
    if model == "xgb":
        import xgboost as xgb

        dtr = xgb.DMatrix(x_tr, label=y_tr)
        dst = xgb.DMatrix(x_stop, label=y_stop)
        m = xgb.train(
            XGB_PARAMS,
            dtr,
            num_boost_round=MAX_ROUNDS,
            evals=[(dst, "stop")],
            early_stopping_rounds=STOPPING,
            verbose_eval=False,
        )
        best = int(m.best_iteration)
        rng = (0, best + 1)
        return (
            m.predict(xgb.DMatrix(x_va), iteration_range=rng),
            m.predict(xgb.DMatrix(x_ts), iteration_range=rng),
            best,
        )
    if model == "cat":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier(
            iterations=MAX_ROUNDS, early_stopping_rounds=STOPPING, **CAT_PARAMS
        )
        m.fit(x_tr, y_tr, eval_set=(x_stop, y_stop))
        return (
            m.predict_proba(x_va)[:, 1],
            m.predict_proba(x_ts)[:, 1],
            int(m.get_best_iteration() or 0),
        )
    raise ValueError(f"unknown model {model!r}")


def run_arm(
    s: Static, name: str, arm: Arm, quiet: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit one arm across the shared folds; return (oof, test_ranks, oof_auc)."""
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    oof = np.zeros(len(s.y))
    test_ranks = np.zeros(len(s.x_test))
    needs_te = "te" in arm.blocks
    # Categorical indices only survive while `raw` is the leading block and is unsliced.
    cat_idx = s.cat_idx if arm.model == "lgb" and arm.blocks[0] == "raw" else []

    for k, (tr, va) in enumerate(folds(s.y)):
        if needs_te:
            te_tr, te_va, te_test = fold_target_encoding(s, tr, va)
        else:
            te_tr = te_va = te_test = None
        x_tr = _assemble(s, arm.blocks, tr, te_tr)
        x_va = _assemble(s, arm.blocks, va, te_va)
        x_ts = _assemble(s, arm.blocks, None, te_test)

        fit_i, stop_i = train_test_split(
            np.arange(len(tr)), test_size=0.1, random_state=SEED + k, stratify=s.y[tr]
        )
        p_va, p_ts, rounds = _fit_predict(
            arm.model,
            x_tr[fit_i],
            s.y[tr][fit_i],
            x_tr[stop_i],
            s.y[tr][stop_i],
            x_va,
            x_ts,
            cat_idx,
        )
        oof[va] = p_va
        # Rank-average across folds: fold models are differently calibrated and AUC only
        # reads order, so averaging ranks is sounder than averaging probabilities.
        test_ranks += rankdata(p_ts) / (N_FOLDS * len(x_ts))
        if not quiet:
            print(
                f"  fold {k} rounds={rounds} auc={roc_auc_score(s.y[va], p_va):.6f}",
                flush=True,
            )

    auc = float(roc_auc_score(s.y, oof))
    if not quiet:
        print(f"{name}: OOF AUC = {auc:.6f}  [{time.time() - t0:.0f}s]", flush=True)
    return oof, test_ranks, auc


def save_member(name: str, oof: np.ndarray, test_ranks: np.ndarray) -> None:
    mem = PROCESSED_DIR / "members"
    mem.mkdir(parents=True, exist_ok=True)
    np.save(mem / f"oof_{name}.npy", oof)
    np.save(mem / f"test_{name}.npy", test_ranks)


def main() -> None:
    requested = list(ARMS)
    for i, a in enumerate(sys.argv):
        if a == "--arms" and i + 1 < len(sys.argv):
            requested = sys.argv[i + 1].split(",")
    unknown = [a for a in requested if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")

    s = prepare(pl.read_csv(TRAIN_CSV), pl.read_csv(TEST_CSV))
    print(f"train={s.x.shape} test={s.x_test.shape} pos_rate={s.y.mean():.4f}\n")
    scores: dict[str, float] = {}
    for name in requested:
        print(f"--- {name} ({'+'.join(ARMS[name].blocks)}, {ARMS[name].model})")
        oof, test_ranks, auc = run_arm(s, name, ARMS[name])
        save_member(name, oof, test_ranks)
        scores[name] = auc

    print("\nmembers written to", PROCESSED_DIR / "members")
    for name, auc in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:11s} {auc:.6f}")


if __name__ == "__main__":
    main()
