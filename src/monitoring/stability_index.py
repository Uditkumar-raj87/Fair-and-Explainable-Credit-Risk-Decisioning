"""Population and characteristic stability indices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _distribution(values: pd.Series | np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    result = counts / max(counts.sum(), 1)
    return np.clip(result, 1e-6, None)


def psi(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, bins: int = 10) -> float:
    """Calculate PSI using shared quantile-derived bins from the expected sample."""
    expected_array = np.asarray(expected, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected_array, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_dist = _distribution(expected_array, edges)
    actual_dist = _distribution(actual_array, edges)
    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))


def csi(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, bins: int = 10) -> float:
    return psi(expected, actual, bins)


def feature_csi(expected: pd.DataFrame, actual: pd.DataFrame, bins: int = 10) -> pd.Series:
    common = [column for column in expected.columns if column in actual.columns]
    return pd.Series({column: csi(expected[column], actual[column], bins) for column in common}, dtype=float)


def stability_alert(value: float) -> str:
    if value > 0.25:
        return "alert"
    if value >= 0.10:
        return "warning"
    return "stable"