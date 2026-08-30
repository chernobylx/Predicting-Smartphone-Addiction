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

**Baseline to beat.** LightGBM on the raw features with no imputation at all (native NaN handling),
5-fold stratified CV: **AUC 0.9580 ± 0.0007**, trained in 24 seconds. Fold-to-fold spread is small
enough that improvements below roughly ±0.002 are not distinguishable from noise at 5 folds.

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

## Results

_Leaderboard results and model summary will be added as the competition progresses._
