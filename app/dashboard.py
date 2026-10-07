"""Streamlit governance dashboard; accepts optional CSV artifacts for review."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.monitoring.stability_index import stability_alert

st.set_page_config(page_title="Credit Risk Governance", layout="wide")
st.title("Credit Risk Governance")
st.warning("EDUCATIONAL USE ONLY: Not for production lending. Does not constitute legal or regulatory compliance.")
uploaded = st.file_uploader("Upload a governance metrics CSV", type="csv")
data = pd.read_csv(uploaded) if uploaded is not None else pd.DataFrame()

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