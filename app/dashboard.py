"""Streamlit governance dashboard; accepts optional CSV artifacts for review."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit can execute this file with app/ as the import root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.stability_index import stability_alert

st.set_page_config(page_title="Credit Risk Governance", layout="wide")
st.title("Credit Risk Governance")
st.warning("EDUCATIONAL USE ONLY: Not for production lending. Does not constitute legal or regulatory compliance.")
uploaded = st.file_uploader("Upload a governance metrics CSV", type="csv")


def preview_artifacts() -> pd.DataFrame:
    models = ["Scorecard"] * 5 + ["Challenger"] * 5
    return pd.DataFrame(
        {
            "model": models,
            "fpr": [0.0, 0.08, 0.18, 0.35, 1.0] * 2,
            "tpr": [0.0, 0.44, 0.68, 0.84, 1.0, 0.0, 0.51, 0.74, 0.9, 1.0],
            "recall": [0.0, 0.44, 0.68, 0.84, 1.0, 0.0, 0.51, 0.74, 0.9, 1.0],
            "precision": [1.0, 0.76, 0.61, 0.48, 0.32, 1.0, 0.79, 0.65, 0.51, 0.33],
            "mean_predicted": [0.05, 0.2, 0.4, 0.65, 0.9] * 2,
            "observed_rate": [0.06, 0.18, 0.38, 0.61, 0.86, 0.05, 0.21, 0.41, 0.64, 0.88],
            "group": ["A", "B", "C", "D", "E"] * 2,
            "approval_rate": [0.72, 0.68, 0.63, 0.59, 0.55] * 2,
            "feature": ["income", "debt_to_income", "credit_history_months", "prior_delinquencies", "score"] * 2,
            "csi": [0.04, 0.13, 0.08, 0.27, 0.11] * 2,
        }
    )


data = pd.read_csv(uploaded) if uploaded is not None else preview_artifacts()
if uploaded is None:
    st.info("Preview mode: charts use synthetic data. Upload a governance CSV to replace it.")

performance, fairness, stability = st.tabs(["Performance & Calibration", "Fairness & Subgroups", "Stability"])
with performance:
    st.subheader("Performance and calibration")
    st.info("Upload a governance artifact with model-specific curve and calibration columns to compare outputs.")
    if not data.empty:
        if {"model", "fpr", "tpr"}.issubset(data.columns):
            st.plotly_chart(px.line(data, x="fpr", y="tpr", color="model", markers=True, title="ROC curves"), use_container_width=True)
        if {"model", "recall", "precision"}.issubset(data.columns):
            st.plotly_chart(px.line(data, x="recall", y="precision", color="model", markers=True, title="Precision-recall curves"), use_container_width=True)
        if {"model", "mean_predicted", "observed_rate"}.issubset(data.columns):
            calibration = data[["model", "mean_predicted", "observed_rate"]].melt(
                id_vars=["model", "mean_predicted"], var_name="measure", value_name="rate"
            )
            st.plotly_chart(px.line(calibration, x="mean_predicted", y="rate", color="model", symbol="measure", markers=True, title="Reliability diagrams"), use_container_width=True)
        st.dataframe(data, use_container_width=True)
with fairness:
    st.subheader("Fairness and subgroups")
    if {"group", "approval_rate"}.issubset(data.columns):
        st.bar_chart(data.set_index("group")["approval_rate"])
    if not data.empty:
        st.dataframe(data, use_container_width=True)
with stability:
    st.subheader("Stability")
    if {"feature", "csi"}.issubset(data.columns):
        display = data[["feature", "csi"]].copy()
        display["status"] = display["csi"].map(stability_alert)
        st.dataframe(display, use_container_width=True)
    else:
        st.info("Upload a metrics artifact with feature and csi columns to review CSI and score PSI.")