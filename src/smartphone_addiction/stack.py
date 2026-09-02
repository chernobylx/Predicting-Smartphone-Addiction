"""Greedy hill-climb rank stack over members, with an honest held-out estimate.

THE TRAP THIS AVOIDS. Choosing blend weights by maximising OOF AUC and then
reporting that same OOF AUC is optimistic: the weights were fitted on the number
being quoted. That is the ensembling-OOF trap, and on a leaderboard where 200
teams span 8e-5 it is the difference between a real gain and a fictional one.

So weights are selected on 4 folds and scored on the held-out 5th, rotating over
all five; the reported figure is the mean of those held-out scores. Only after
that are weights refit on all rows to build the submission.

Selection is greedy forward with replacement (Caruana et al. 2004): a member may
be chosen repeatedly, which produces integer weights without a continuous
optimiser and is markedly harder to overfit than least-squares blending.

WHAT EARNED WEIGHT, measured over 29 candidate members:
  - Feature-SUBSET members did (struct_imp 0.100, no_te 0.033). Removing a whole
    information channel forces a genuinely different solution.
  - A different model CLASS did (embedding NN, 0.100 across two variants) at 0.96
    correlation, versus 0.995+ among the tree members.
  - Hyperparameter diversity did NOT: linear_tree, extra_trees, DART and ExtraTrees all
    earned exactly zero. Same matrix, same solution, no independent error to cancel.

Run:  pixi run python -m smartphone_addiction.stack [--submit]
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from smartphone_addiction.paths import (
    PROCESSED_DIR,
    SUBMISSIONS_DIR,
    TEST_CSV,
    TRAIN_CSV,
)

SEED = 42
ROUNDS = 30  # greedy rounds; weights land in multiples of 1/30
OUTER_SEED = 3  # deliberately not SEED: the weight-selection split is its own thing
RESOLUTION_FLOOR = 0.00014  # measured paired sd on a public-split-sized bootstrap


def load_members() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load every member that has BOTH an OOF and a test vector, as normalised ranks."""
    y = pl.read_csv(TRAIN_CSV)["addicted_label"].to_numpy()
    mem_dir = PROCESSED_DIR / "members"
    names, oofs, tests = [], [], []
    for path in sorted(mem_dir.glob("oof_*.npy")):
        name = path.stem[4:]
        test_path = mem_dir / f"test_{name}.npy"
        if not test_path.exists():
            print(f"  skip {name}: no test vector")
            continue
        oof = np.load(path)
        if len(oof) != len(y):
            print(f"  skip {name}: length {len(oof)} != {len(y)}")
            continue
        test = np.load(test_path)
        names.append(name)
        oofs.append(rankdata(oof) / len(oof))
        tests.append(rankdata(test) / len(test))
    return names, np.array(oofs), np.array(tests), y


def hillclimb(members: np.ndarray, y: np.ndarray, rounds: int = ROUNDS) -> np.ndarray:
    """Greedy forward selection with replacement; returns normalised weights."""
    weights = np.zeros(len(members))
    current = np.zeros(len(y))
    for r in range(rounds):
        best_j, best_score = 0, -1.0
        for j in range(len(members)):
            score = roc_auc_score(y, (current * r + members[j]) / (r + 1))
            if score > best_score:
                best_score, best_j = score, j
        weights[best_j] += 1
        current = (current * r + members[best_j]) / (r + 1)
    return weights / weights.sum()


def main() -> None:
    names, oofs, tests, y = load_members()
    print(f"{len(names)} members: {names}\n")
    solo = [roc_auc_score(y, oofs[i]) for i in range(len(names))]
    for name, score in sorted(zip(names, solo, strict=True), key=lambda kv: -kv[1]):
        print(f"  {name:11s} solo_auc={score:.6f}")

    # Honest estimate: weights fitted on 4 folds, scored on the 5th.
    held = []
    outer = StratifiedKFold(5, shuffle=True, random_state=OUTER_SEED)
    for tr, va in outer.split(np.zeros(len(y)), y):
        w = hillclimb(oofs[:, tr], y[tr])
        held.append(roc_auc_score(y[va], w @ oofs[:, va]))
    held_mean = float(np.mean(held))
    best_solo = max(solo)
    gain = held_mean - best_solo
    print(f"\nHONEST held-out stack AUC = {held_mean:.6f} (folds {np.round(held, 5)})")
    best_name = names[int(np.argmax(solo))]
    print(f"best solo member          = {best_solo:.6f}  ({best_name})")
    print(f"honest stack gain         = {gain:+.6f}")

    weights = hillclimb(oofs, y)
    print("\nfinal weights:")
    for name, w in sorted(zip(names, weights, strict=True), key=lambda kv: -kv[1]):
        if w > 0:
            print(f"  {name:11s} {w:.3f}")

    if "--submit" not in sys.argv:
        return
    if gain <= RESOLUTION_FLOOR:
        print(f"\ngain {gain:+.6f} is below the {RESOLUTION_FLOOR} resolution floor "
              "— no file written")
        return
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / "lgbm_stack.csv"
    ids = pl.read_csv(TEST_CSV)["id"]
    pl.DataFrame({"id": ids, "addicted_label": weights @ tests}).write_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
