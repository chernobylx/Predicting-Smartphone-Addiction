# Predicting Smartphone Addiction

Machine learning entry for the Kaggle [Playground Series — Season 6, Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8) competition: predicting the likelihood of smartphone addiction from behavioral and usage data.

## The problem

| | |
|---|---|
| **Task** | Binary classification — predict `addicted_label` |
| **Metric** | **ROC-AUC** |
| **Training data** | 691,369 rows × 12 features (9 numeric, 3 categorical) |
| **Class balance** | 70.9% positive |
| **Missingness** | Every feature 4–20% null; 61.1% of rows have at least one null |

Because the metric is AUC, only the *ranking* of predictions matters: probability calibration and
decision-threshold tuning are irrelevant to the score, and rank-based ensembling is well motivated.

## Results

| Model | CV (OOF AUC) | Public LB |
|---|---|---|
| [`baseline.py`](src/smartphone_addiction/baseline.py) — raw features, no imputation | 0.963947 | 0.96541 |
| [`pipeline.py`](src/smartphone_addiction/pipeline.py) — full representation pipeline | **0.968069** | _pending_ |

The +0.0044 gain is **representation, not tuning**. It decomposes into three mechanisms,
each measured as a paired arm on identical folds:

| Mechanism | Gain | Why it works |
|---|---|---|
| `max_bin` 255 → 2047 | +0.00215 | Six of nine numeric columns have more distinct values than LightGBM's default 255 bins (`weekend_screen_time` 1460, `daily_screen_time_hours` 1398). The label rule has hard steps at exact 2-decimal values; at 255 bins the bucket straddling `daily = 8.00` holds ~2,300 rows the model cannot separate. |
| Value-level encoding | +0.00122 | `notifications_per_day` has raw univariate AUC 0.5079 but **0.7486** target-encoded. Adjacent integer values differ in target rate by 0.22 against a sampling sd of 0.008 — the columns are lookup keys, not quantities, and a tree needs two splits per value to express that. |
| Generator structure | +0.00105 | `daily ≥ social + gaming + work` holds for **100.000%** of complete rows, so the residual is a real hidden quantity and, where a term is missing, the constraint *bounds* it. A four-term linear combination is the one thing axis-aligned trees provably cannot construct. |

**Measured and rejected**, each as a paired same-fold comparison:

| Idea | Effect |
|---|---|
| Hyperparameter search (`num_leaves` 31/63/255) | span 0.00046, less than one fold sd |
| Capacity re-tune on the wide matrix (127 leaves) | +0.00001 |
| Appending `original.csv` to training | −0.00004 (4/5 folds worse) |
| LightGBM + XGBoost blend | +0.00011 (rank correlation 0.99542) |
| Ordinal transfer feature from `original.csv` | −0.00024 (5/5 folds worse) |
| Missingness indicators / null counts | univariate AUC 0.5017 — MCAR, inert |

**Imputation strategy: none.** LightGBM's learned per-split default direction for NaN is
strictly more expressive than a point estimate. Missingness is handled instead by giving NaN
its own encoding level, computing the budget residual under both NaN conventions, and
deriving hard bounds from the generator constraint where the algebra pins a missing value down.

## Setup

```bash
# Install pixi if needed: https://pixi.sh
pixi install
```

## Data

Competition data is not committed to the repo (per Kaggle rules). To download it you need a Kaggle API token:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`)
3. Accept the competition rules on the [competition page](https://www.kaggle.com/competitions/playground-series-s6e8)

Then:

```bash
pixi run data
```

This downloads and extracts `train.csv`, `test.csv`, and `sample_submission.csv` into `data/raw/`.

## Project Structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original competition data (gitignored, rebuild with `pixi run data`) |
| `data/processed/` | Cleaned and transformed outputs |
| `notebooks/` | Exploratory analysis (marimo apps) |
| `src/smartphone_addiction/` | Reusable Python package (features, models, submission helpers) |
| `submissions/` | Generated submission files (gitignored) |
| `tests/` | Unit tests |
| `docs/figures/` | Exported figures referenced in write-ups |

## Development

```bash
pixi run check       # lint + typecheck + test
pixi run format      # auto-format with ruff
pixi run lint        # check style with ruff
pixi run typecheck   # check types with ty
pixi run test        # run pytest
```

Notebooks are [marimo](https://marimo.io) apps — plain `.py` files that diff cleanly. Open one with:

```bash
pixi run marimo edit notebooks/<name>.py
```

## Reproducing the results

```bash
pixi run python -m smartphone_addiction.baseline   # raw-feature baseline  -> submissions/lgbm_baseline.csv
pixi run python -m smartphone_addiction.pipeline   # full pipeline         -> submissions/lgbm_pipeline.csv
```

Both are deterministic at seed 42 and write out-of-fold predictions to `data/processed/`
so any later comparison can be run as a paired same-fold test.
