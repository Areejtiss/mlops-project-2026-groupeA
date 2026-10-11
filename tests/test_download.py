"""Tests for the Kaggle download module (no network, no real credentials)."""

from pathlib import Path

import pytest

from crediscore.data import download

CREDENTIAL_VARS = ["KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"]


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every Kaggle credential from the environment."""
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)


def test_missing_credentials_gives_clear_error(
    tmp_path: Path, no_credentials: None
) -> None:
    with pytest.raises(download.KaggleDownloadError, match="credentials not found"):
        download.download(tmp_path)


def test_legacy_credentials_need_username_and_key(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    assert not download.has_kaggle_credentials()
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    assert download.has_kaggle_credentials()


def test_token_alone_is_enough(
    monkeypatch: pytest.MonkeyPatch, no_credentials: None
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    assert download.has_kaggle_credentials()


def test_skips_download_when_files_exist(tmp_path: Path, no_credentials: None) -> None:
    for name in download.FILES:
        (tmp_path / name).write_text("x")
    # No credentials and no network: it must return without trying to download.
    assert download.download(tmp_path) == []


def test_unzip_if_needed_extracts_and_removes_archive(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "application_train.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("application_train.csv", "a,b\n1,2\n")
    download.unzip_if_needed(tmp_path, "application_train.csv")
    assert (tmp_path / "application_train.csv").exists()
    assert not archive.exists()


def test_check_shape_rejects_wrong_file(tmp_path: Path) -> None:
    (tmp_path / "application_train.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unexpected shape"):
        download.check_application_train(tmp_path)
