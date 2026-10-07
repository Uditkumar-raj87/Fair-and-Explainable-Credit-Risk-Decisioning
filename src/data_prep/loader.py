"""Load educational credit-risk data without hiding provenance or warnings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)
EDUCATIONAL_WARNING: Final[str] = (
    "EDUCATIONAL USE ONLY: Not for production lending. "
    "Does not constitute legal or regulatory compliance."
)
GERMAN_CREDIT_URL: Final[str] = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def _warn_educational_use() -> None:
    LOGGER.warning(EDUCATIONAL_WARNING)


def load_credit_data(path: str | Path | None = None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load a local CSV or a supplied frame; no network download occurs implicitly.

    The German Credit dataset is a supported educational source. A local copy should
    be downloaded and reviewed by the user before passing its path to this function.
    """
    _warn_educational_use()
    if frame is not None:
        return frame.copy()
    if path is None:
        raise ValueError("Provide a local CSV path or a pandas DataFrame.")
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def ensure_datetime_column(
    frame: pd.DataFrame, date_column: str = "application_date"
) -> pd.DataFrame:
    """Return a copy with a usable date column, simulating dates when absent."""
    result = frame.copy()
    if date_column not in result:
        result[date_column] = pd.date_range("2020-01-01", periods=len(result), freq="D")
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    return result