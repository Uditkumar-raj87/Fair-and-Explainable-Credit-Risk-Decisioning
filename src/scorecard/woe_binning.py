"""Small, auditable WoE and IV transformer for educational scorecards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class WOETransformer:
    bins: int = 10
    smoothing: float = 0.5
    mappings: dict[str, dict[Any, float]] = field(default_factory=dict)
    edges: dict[str, np.ndarray] = field(default_factory=dict)
    iv_: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        y_values = pd.Series(y, index=X.index).astype(int)
        total_good = max(int((y_values == 0).sum()), 1)
        total_bad = max(int((y_values == 1).sum()), 1)
        for feature in X.columns:
            series = X[feature]
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > self.bins:
                _, edges = pd.qcut(series, q=self.bins, duplicates="drop", retbins=True)
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges[feature] = edges
                grouped = pd.cut(series, bins=edges, include_lowest=True)
            else:
                grouped = series.astype("object").where(series.notna(), "__MISSING__")
            stats = pd.DataFrame({"bin": grouped, "target": y_values}).groupby("bin", observed=False)["target"].agg(["count", "sum"])
            stats["good"] = stats["count"] - stats["sum"]
            stats["dist_good"] = (stats["good"] + self.smoothing) / (total_good + self.smoothing * len(stats))
            stats["dist_bad"] = (stats["sum"] + self.smoothing) / (total_bad + self.smoothing * len(stats))
            stats["woe"] = np.log(stats["dist_good"] / stats["dist_bad"])
            stats["iv"] = (stats["dist_good"] - stats["dist_bad"]) * stats["woe"]
            self.mappings[feature] = stats["woe"].to_dict()
            self.iv_[feature] = float(stats["iv"].sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for feature, mapping in self.mappings.items():
            series = X[feature]
            if feature in self.edges:
                values = pd.cut(series, bins=self.edges[feature], include_lowest=True)
            else:
                values = series.astype("object").where(series.notna(), "__MISSING__")
            result[feature] = values.map(mapping).fillna(0.0).astype(float)
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def filter_features(self, threshold: float = 0.02) -> list[str]:
        return [feature for feature, iv in self.iv_.items() if iv >= threshold]


def calculate_woe_iv(X: pd.DataFrame, y: pd.Series, bins: int = 10) -> tuple[pd.DataFrame, dict[str, float], WOETransformer]:
    """Return WoE values, IV by feature, and the fitted transformer."""
    transformer = WOETransformer(bins=bins)
    transformed = transformer.fit_transform(X, y)
    return transformed, transformer.iv_.copy(), transformer


def monotonic_bin_order(transformer: WOETransformer, feature: str) -> list[Any]:
    """Return bins ordered by WoE; this is an auditable monotonic-risk proxy."""
    return [key for key, _ in sorted(transformer.mappings.get(feature, {}).items(), key=lambda item: item[1])]