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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMChallenger":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:, 1]