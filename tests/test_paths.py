"""Sanity checks for the project layout."""

from smartphone_addiction import paths


def test_project_root_is_repo_root():
    assert (paths.PROJECT_ROOT / "pixi.toml").exists()


def test_data_dirs_exist():
    assert paths.RAW_DIR.is_dir()
    assert paths.PROCESSED_DIR.is_dir()
    assert paths.SUBMISSIONS_DIR.is_dir()
