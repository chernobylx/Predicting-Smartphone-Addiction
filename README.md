# Predicting Smartphone Addiction

Machine learning entry for the Kaggle [Playground Series — Season 6, Episode 8](https://www.kaggle.com/competitions/playground-series-s6e8) competition: predicting the likelihood of smartphone addiction from behavioral and usage data.

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
