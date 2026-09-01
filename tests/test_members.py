"""Adversarial checks on the representation layer, run on synthetic data.

The competition CSVs are not committed (Kaggle rules), so these tests build a small
frame with the same schema and the same generator constraint. That is enough to exercise
every code path in `features.py` and `members.py`, and — more importantly — to test the
one claim this project rests on: that the target encoder does not leak.

The leakage test is the battery TASKS.md asked for and never got. Under a shuffled label
the whole pipeline must score chance. A leaking encoder cannot.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from smartphone_addiction import features, members

N_TRAIN, N_TEST = 900, 300
NUMERIC = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time",
    "notifications_per_day",
    "age",
    "apps_installed",
]


def _synth(
    n: int, rng: np.random.Generator, *, interleave: bool = False
) -> pl.DataFrame:
    """The real schema: the generator constraint, NaNs, and 3 categoricals."""
    social = np.round(rng.uniform(0, 4, n), 2)
    gaming = np.round(rng.uniform(0, 3, n), 2)
    work = np.round(rng.uniform(0, 5, n), 2)
    other = np.round(rng.uniform(0, 2, n), 2)
    # The constraint the ledger found: daily >= social + gaming + work, exactly.
    daily = np.round(social + gaming + work + other, 2)
    cols = {
        "daily_screen_time_hours": daily,
        "social_media_hours": social,
        "gaming_hours": gaming,
        "work_study_hours": work,
        "sleep_hours": np.round(rng.uniform(3, 10, n), 2),
        "weekend_screen_time": np.round(daily * rng.uniform(0.8, 1.4, n), 2),
        "notifications_per_day": rng.integers(0, 300, n).astype(float),
        "age": rng.integers(13, 60, n).astype(float),
        "apps_installed": rng.integers(5, 120, n).astype(float),
        "gender": rng.choice(["Female", "Male", "Other"], n),
        "stress_level": rng.choice(["Low", "Medium", "High"], n),
        "academic_work_impact": rng.choice(["No", "Yes"], n),
    }
    # Every feature is 4-20% null in the real data; missingness is the norm here.
    for c in NUMERIC:
        mask = rng.random(n) < 0.10
        cols[c] = np.where(mask, np.nan, cols[c])
    order = list(cols)
    if (
        interleave
    ):  # categoricals in the middle: the layout the old positional slice broke on
        order = (
            order[:3]
            + ["gender"]
            + order[3:9]
            + ["stress_level", "academic_work_impact"]
        )
    frame = {"id": np.arange(n)} | {c: cols[c] for c in order}
    return pl.DataFrame(frame)


def _labelled(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """A learnable label: a hard step on daily, as the real generator has."""
    daily = df["daily_screen_time_hours"].to_numpy()
    logit = np.where(np.isnan(daily), 0.0, (daily - 6.0)) + rng.normal(0, 0.5, len(df))
    y = (logit > 0).astype(np.int64)
    return df.with_columns(pl.Series("addicted_label", y))


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def static(rng) -> features.Static:
    train = _labelled(_synth(N_TRAIN, rng), rng)
    test = _synth(N_TEST, rng)
    return features.prepare(train, test)


def test_prepare_block_widths(static):
    n_feats = len(static.feats)
    assert n_feats == 12, static.feats
    assert static.n_num == 9
    assert static.freq.shape == (N_TRAIN, n_feats)
    assert static.keys.shape == (N_TRAIN, 9)
    assert static.struct.shape[0] == N_TRAIN
    assert static.struct_test.shape == (N_TEST, static.struct.shape[1])


def test_numeric_keys_survive_interleaved_columns(rng):
    """Keys are selected by name, so a categorical in the middle must not shift them."""
    plain = features.prepare(_labelled(_synth(200, rng), rng), _synth(50, rng))
    mixed = features.prepare(
        _labelled(_synth(200, np.random.default_rng(0)), np.random.default_rng(0)),
        _synth(50, np.random.default_rng(0)),
    )
    # Whatever the column order, the key block covers the 9 numeric columns
    # and nothing else.
    assert plain.keys.shape[1] == mixed.keys.shape[1] == 9
    inter = features.prepare(
        _labelled(
            _synth(200, np.random.default_rng(1), interleave=True),
            np.random.default_rng(1),
        ),
        _synth(50, np.random.default_rng(1), interleave=True),
    )
    assert inter.keys.shape[1] == 9
    # A categorical never enters the key block: its codes are 0/1/2, so a column of
    # keys drawn from one would have at most 3 distinct non-missing values.
    distinct = [len(np.unique(inter.keys[:, j])) for j in range(9)]
    assert min(distinct) > 3, distinct


def test_folds_are_identical_across_arms(static):
    """Paired comparison is only valid if every arm sees the same split."""
    a = [(tuple(tr), tuple(va)) for tr, va in features.folds(static.y)]
    b = [(tuple(tr), tuple(va)) for tr, va in features.folds(static.y)]
    assert a == b
    covered = np.concatenate([np.array(va) for _, va in a])
    assert sorted(covered.tolist()) == list(range(len(static.y)))


def test_target_encoding_never_encodes_a_row_from_itself(static):
    """The inner nesting: a training row's encoding must not depend on its own label."""
    tr, va = next(features.folds(static.y))
    te_tr, te_va, te_test = features.fold_target_encoding(static, tr, va)
    assert te_tr.shape == (len(tr), static.n_num)
    assert te_va.shape == (len(va), static.n_num)
    assert te_test.shape == (len(static.keys_test), static.n_num)
    assert np.isfinite(te_tr).all() and np.isfinite(te_va).all()
    # Flip one training row's label and re-encode. The encoding must respond:
    # an encoder that ignores the label entirely would pass every other check.
    flipped = features.Static(**{**static.__dict__, "y": static.y.copy()})
    flipped.y[tr[0]] ^= 1
    te_tr2, _, _ = features.fold_target_encoding(flipped, tr, va)
    assert not np.allclose(te_tr, te_tr2), (
        "flipping a label changed nothing — encoder is inert"
    )


@pytest.mark.parametrize("arm_name", ["full", "no_te", "struct_imp", "raw"])
def test_arm_learns_a_real_signal(static, arm_name, monkeypatch):
    monkeypatch.setattr(members, "MAX_ROUNDS", 40)
    monkeypatch.setattr(members, "STOPPING", 10)
    oof, test_ranks, auc = members.run_arm(
        static, arm_name, members.ARMS[arm_name], quiet=True
    )
    assert oof.shape == (N_TRAIN,)
    assert test_ranks.shape == (N_TEST,)
    assert np.isfinite(oof).all() and np.isfinite(test_ranks).all()
    assert auc > 0.75, f"{arm_name} failed to learn a planted signal: {auc:.4f}"


@pytest.mark.parametrize("model", ["xgb", "cat"])
def test_other_model_classes_run(static, model, monkeypatch):
    """Model-class diversity also earned stack weight; keep those paths working."""
    monkeypatch.setattr(members, "MAX_ROUNDS", 40)
    monkeypatch.setattr(members, "STOPPING", 10)
    name = f"{model}_full"
    _, _, auc = members.run_arm(static, name, members.ARMS[name], quiet=True)
    assert auc > 0.75, f"{name}: {auc:.4f}"


def test_pipeline_does_not_leak_under_a_shuffled_label(rng, monkeypatch):
    """THE test. Random labels, full target-encoding path, must score chance.

    A target encoder that lets a row see its own label scores far above 0.5 here, and
    that is exactly the failure that produces a CV gain with no leaderboard movement.
    """
    monkeypatch.setattr(members, "MAX_ROUNDS", 60)
    monkeypatch.setattr(members, "STOPPING", 10)
    train = _synth(N_TRAIN, rng)
    noise = np.random.default_rng(7).integers(0, 2, N_TRAIN)
    train = train.with_columns(pl.Series("addicted_label", noise))
    s = features.prepare(train, _synth(N_TEST, rng))
    oof, _, auc = members.run_arm(s, "full", members.ARMS["full"], quiet=True)
    assert abs(auc - 0.5) < 0.05, (
        f"OOF AUC {auc:.4f} on random labels — the encoder leaks"
    )
    assert abs(roc_auc_score(s.y, oof) - auc) < 1e-12


def test_hillclimb_concentrates_weight_on_the_informative_member():
    """`stack.hillclimb` is pure; check it prefers signal to noise without any data."""
    from smartphone_addiction.stack import hillclimb

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.3, 400)
    noise1, noise2 = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    from scipy.stats import rankdata

    mem = np.array([rankdata(m) / len(m) for m in (good, noise1, noise2)])
    w = hillclimb(mem, y, rounds=20)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > 0.7, w
