"""Protected-attribute slices for educational fairness review."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import error_rates, selection_rate


def subgroup_report(
    frame: pd.DataFrame,
    group_column: str,
    y_true_column: str,
    probability_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Report approval rate, FPR, and FNR for each observed group."""
    rows: list[dict[str, object]] = []
    for group, subset in frame.groupby(group_column, dropna=False):
        y_true = subset[y_true_column].to_numpy().astype(int)
        approved = (subset[probability_column].to_numpy() < threshold).astype(int)
        rates = error_rates(y_true, 1 - approved)
        rows.append({"group": group, "approval_rate": selection_rate(approved), **rates, "count": len(subset)})
    return pd.DataFrame(rows)


def fairness_gaps(report: pd.DataFrame) -> dict[str, float]:
    if report.empty:
        return {"approval_rate_gap": 0.0, "fpr_gap": 0.0, "fnr_gap": 0.0}
    return {
        "approval_rate_gap": float(report["approval_rate"].max() - report["approval_rate"].min()),
        "fpr_gap": float(report["fpr"].max() - report["fpr"].min()),
        "fnr_gap": float(report["fnr"].max() - report["fnr"].min()),
    }