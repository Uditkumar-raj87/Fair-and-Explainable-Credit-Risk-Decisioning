import numpy as np
import pandas as pd

from src.fairness.metrics import optimize_threshold
from src.fairness.subgroup_analysis import fairness_gaps, subgroup_report
from src.monitoring.stability_index import feature_csi, stability_alert


def test_threshold_optimization_returns_valid_cutoff():
    result = optimize_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8]))
    assert 0 < result["threshold"] < 1


def test_subgroup_report_and_gaps():
    frame = pd.DataFrame({"group": ["A", "A", "B", "B"], "y": [0, 1, 0, 1], "p": [0.1, 0.8, 0.2, 0.9]})
    report = subgroup_report(frame, "group", "y", "p", 0.5)
    assert set(report["group"]) == {"A", "B"}
    assert fairness_gaps(report)["approval_rate_gap"] == 0


def test_stability_alert_bands():
    assert stability_alert(0.05) == "stable"
    assert stability_alert(0.10) == "warning"
    assert stability_alert(0.26) == "alert"
    values = feature_csi(pd.DataFrame({"x": [1, 2, 3]}), pd.DataFrame({"x": [1, 2, 4]}))
    assert "x" in values