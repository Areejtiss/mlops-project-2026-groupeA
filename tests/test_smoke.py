"""Smoke test: the package is installed and importable."""

import crediscore


def test_package_imports() -> None:
    assert crediscore.__version__ == "0.1.0"
