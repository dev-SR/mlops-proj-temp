"""Thin Streamlit UI that displays the latest Evidently drift report."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.shoppers import config, drift

st.set_page_config(page_title="Shoppers Drift Dashboard", layout="wide")
st.title("Online Shoppers — Evidently Drift Dashboard")
st.caption("Thin UI layer; report generation lives in src.shoppers.drift")

report_path = config.REPORTS_DIR / "drift_report.html"

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate / refresh drift report", type="primary"):
        try:
            path = drift.build_drift_report()
            st.success(f"Report written to {path}")
            report_path = path
        except Exception as exc:  # noqa: BLE001 - show errors in UI
            st.error(str(exc))

with col2:
    st.write(f"Looking for: `{report_path}`")

if report_path.exists():
    html = Path(report_path).read_text(encoding="utf-8")
    st.components.v1.html(html, height=900, scrolling=True)
else:
    st.info(
        "No drift report yet. Make some `/predict` calls, then click "
        "**Generate / refresh drift report**."
    )

if config.METRICS_PATH.exists():
    st.subheader("Latest evaluation metrics")
    st.json(config.METRICS_PATH.read_text(encoding="utf-8"))
