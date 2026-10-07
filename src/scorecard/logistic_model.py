"""Interpretable logistic scorecard and adverse-action reason codes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .woe_binning import WOETransformer


def log_odds_to_score(log_odds: float | np.ndarray, target_score: float = 600, target_odds: float = 50, pdo: float = 20) -> float | np.ndarray:
    """Map good:bad odds (log odds) to the conventional score scale."""
    factor = pdo / np.log(2)
    return target_score + factor * (np.asarray(log_odds) - np.log(target_odds))


def score_to_probability(score: float | np.ndarray, target_score: float = 600, target_odds: float = 50, pdo: float = 20) -> float | np.ndarray:
    factor = pdo / np.log(2)
    return 1 / (1 + target_odds * np.exp((np.asarray(score) - target_score) / factor))


@dataclass
class LogisticScorecard:
    iv_threshold: float = 0.02
    transformer: WOETransformer | None = None
    model: LogisticRegression | None = None
    features_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticScorecard":
        self.transformer = WOETransformer().fit(X, y)
        self.features_ = self.transformer.filter_features(self.iv_threshold)
        if not self.features_:
            self.features_ = list(X.columns)
        transformed = self.transformer.transform(X)[self.features_]
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(transformed, y)
        return self

    def _transformed(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None or self.model is None or self.features_ is None:
            raise RuntimeError("Fit the scorecard before prediction")
        return self.transformer.transform(X)[self.features_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._transformed(X))[:, 1]  # type: ignore[union-attr]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        return np.asarray(log_odds_to_score(np.log((1 - probabilities) / probabilities)))

    def reason_codes(self, X: pd.DataFrame, top_n: int = 4) -> list[list[str]]:
        transformed = self._transformed(X)
        coefficients = self.model.coef_[0]  # type: ignore[union-attr]
        rows: list[list[str]] = []
        for _, values in transformed.iterrows():
            deductions = np.maximum(0, -(values.to_numpy() * coefficients))
            indices = np.argsort(deductions)[::-1][:top_n]
            rows.append([f"{self.features_[index]} contributed to score reduction" for index in indices if deductions[index] > 0])
        return rows