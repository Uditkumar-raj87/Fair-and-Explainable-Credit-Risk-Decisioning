"""Calibration methods and comparable model metrics."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

LOGGER = logging.getLogger(__name__)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(total)


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "gini": float(2 * roc_auc_score(y_true, probabilities) - 1),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def platt_scaling(estimator, X_calibration, y_calibration):
    return CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit").fit(X_calibration, y_calibration)


def isotonic_scaling(probabilities: np.ndarray, y_calibration: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(probabilities, y_calibration)


def log_calibration_comparison(y_true: np.ndarray, uncalibrated: np.ndarray, calibrated: np.ndarray) -> dict[str, dict[str, float]]:
    result = {"uncalibrated": calibration_metrics(y_true, uncalibrated), "calibrated": calibration_metrics(y_true, calibrated)}
    LOGGER.info("Calibration comparison: %s", result)
    return result