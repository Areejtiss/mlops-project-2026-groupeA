"""Download the Home Credit raw files from the Kaggle competition.

Usage:
    poetry run python -m crediscore.data.download

Credentials are read from environment variables (loaded from .env if present):
    KAGGLE_API_TOKEN                    (new token format), or
    KAGGLE_USERNAME + KAGGLE_KEY        (legacy kaggle.json format)

The competition rules must be accepted once on kaggle.com, otherwise Kaggle
answers 403 Forbidden.
"""

import os
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

COMPETITION = "home-credit-default-risk"
FILES = ["application_train.csv", "HomeCredit_columns_description.csv"]
RAW_DIR = Path("data/raw")

EXPECTED_ROWS = 307_511
EXPECTED_COLS = 122


class KaggleDownloadError(RuntimeError):
    """Raised when the Kaggle download cannot be done (credentials, rules...)."""


def has_kaggle_credentials() -> bool:
    """Return True if Kaggle credentials are set in the environment."""
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_legacy = bool(os.environ.get("KAGGLE_USERNAME")) and bool(
        os.environ.get("KAGGLE_KEY")
    )
    return has_token or has_legacy


def missing_files(raw_dir: Path = RAW_DIR) -> list[str]:
    """Return the expected files that are not yet in ``raw_dir``."""
    return [name for name in FILES if not (raw_dir / name).exists()]


def unzip_if_needed(raw_dir: Path, file_name: str) -> None:
    """Kaggle may serve large files zipped: extract ``<file>.zip`` then delete it."""
    archive = raw_dir / f"{file_name}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()


def download(raw_dir: Path = RAW_DIR) -> list[str]:
    """Download the missing competition files into ``raw_dir``.

    The function is idempotent: files already present are not downloaded again.

    Args:
        raw_dir: Destination directory.

    Returns:
        Names of the files that were downloaded (empty if nothing was missing).

    Raises:
        KaggleDownloadError: If credentials are missing, or Kaggle refuses access.
    """
    to_download = missing_files(raw_dir)
    if not to_download:
        print(f"All files already in {raw_dir}, nothing to download.")
        return []

    if not has_kaggle_credentials():
        raise KaggleDownloadError(
            "Kaggle credentials not found. Set KAGGLE_API_TOKEN, or KAGGLE_USERNAME "
            "and KAGGLE_KEY, in your .env file (see .env.example)."
        )

    # Imported here on purpose: `import kaggle` tries to authenticate immediately,
    # which would break any code (or test) that imports this module without
    # credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    raw_dir.mkdir(parents=True, exist_ok=True)

    for file_name in to_download:
        print(f"Downloading {file_name} ...")
        try:
            api.competition_download_file(COMPETITION, file_name, path=str(raw_dir))
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise KaggleDownloadError(
                    f"Kaggle refused access ({status}). Check your credentials and "
                    "accept the competition rules at "
                    f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
                ) from exc
            raise
        unzip_if_needed(raw_dir, file_name)

    return to_download


def check_application_train(raw_dir: Path = RAW_DIR) -> tuple[int, int]:
    """Check that application_train.csv has the expected shape.

    Returns:
        The (rows, columns) shape of the file.

    Raises:
        ValueError: If the shape is not 307,511 x 122.
    """
    df = pd.read_csv(raw_dir / "application_train.csv")
    rows, cols = df.shape
    if (rows, cols) != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Unexpected shape {rows} x {cols}, "
            f"expected {EXPECTED_ROWS} x {EXPECTED_COLS}"
        )
    return rows, cols


def main() -> None:
    """Command-line entry point."""
    load_dotenv()  # read .env into environment variables (does not override)
    download()
    rows, cols = check_application_train()
    print(f"OK: application_train.csv has {rows:,} rows x {cols} columns.")


if __name__ == "__main__":
    main()
