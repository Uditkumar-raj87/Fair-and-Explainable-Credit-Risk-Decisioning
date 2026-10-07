"""LightGBM challenger wrapper."""

from __future__ import annotations

import pandas as pd


class LightGBMChallenger:
    def __init__(self, **params: object) -> None:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("Install lightgbm to train the challenger model") from exc
        defaults = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31, "random_state": 42, "verbosity": -1}
        defaults.update(params)
        self.model = LGBMClassifier(**defaults)
        self.feature_columns: list[str] = []

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        encoded = pd.get_dummies(X, dummy_na=True)
        if fit:
            self.feature_columns = list(encoded.columns)
        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMChallenger":
        self.model.fit(self._encode(X, fit=True), y)
        return self

    def predict_proba(self, X: pd.DataFrame):
        if not self.feature_columns:
            raise RuntimeError("Fit the challenger before prediction")
        return self.model.predict_proba(self._encode(X))[:, 1]