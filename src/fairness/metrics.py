"""Decision economics and group fairness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def expected_value(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
    good_loan_profit: float = 100.0,
    default_loss: float = 500.0,
) -> float:
    """Expected portfolio value where probability below threshold is approved."""
    y = np.asarray(y_true).astype(int)
    approved = np.asarray(probabilities) < threshold
    values = np.where(approved & (y == 0), good_loan_profit, 0.0)
    values = np.where(approved & (y == 1), -default_loss, values)
    return float(values.mean())


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
    good_loan_profit: float = 100.0,
    default_loss: float = 500.0,
) -> dict[str, float]:
    values = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    scores = [expected_value(y_true, probabilities, threshold, good_loan_profit, default_loss) for threshold in values]
    index = int(np.argmax(scores))
    return {"threshold": float(values[index]), "expected_value": float(scores[index])}


def error_rates(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {"fpr": float(fp / max(fp + tn, 1)), "fnr": float(fn / max(fn + tp, 1))}


def selection_rate(y_pred: np.ndarray, approval_label: int = 1) -> float:
    return float(np.mean(np.asarray(y_pred) == approval_label))