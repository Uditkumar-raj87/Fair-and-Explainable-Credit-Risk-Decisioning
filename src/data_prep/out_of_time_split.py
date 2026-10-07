"""Out-of-time validation split."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OOTSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


def out_of_time_split(
    frame: pd.DataFrame,
    date_column: str = "application_date",
    test_fraction: float = 0.2,
    cutoff: str | pd.Timestamp | None = None,
) -> OOTSplit:
    """Split chronologically so every test row is later than every train row."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if date_column not in frame:
        raise KeyError(f"Missing date column: {date_column}")
    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="raise")
    ordered = ordered.sort_values(date_column, kind="stable")
    if ordered[date_column].nunique() < 2:
        raise ValueError("At least two distinct dates are required for an OOT split")
    split_index = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction))))
    resolved_cutoff = pd.Timestamp(cutoff) if cutoff is not None else ordered.iloc[split_index][date_column]
    train = ordered[ordered[date_column] < resolved_cutoff].copy()
    test = ordered[ordered[date_column] >= resolved_cutoff].copy()
    if train.empty or test.empty or train[date_column].max() >= test[date_column].min():
        raise ValueError("OOT split must contain non-empty, strictly ordered train and test periods")
    return OOTSplit(train=train, test=test, cutoff=resolved_cutoff)