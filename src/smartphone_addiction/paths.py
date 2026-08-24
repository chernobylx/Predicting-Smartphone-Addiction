"""Project path constants, resolved relative to the repo root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"
