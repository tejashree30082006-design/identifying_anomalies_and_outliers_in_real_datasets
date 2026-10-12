"""
Probability Distribution Analysis page (Module D).
Renders empirical histograms, fitted normal curve overlays (as estimates),
boxplots, and normal Q-Q probability plots.
"""

import streamlit as st
import numpy as np

from core.distributions import analyze_distribution
from core.outlier_detection import detect_iqr_outliers
from utils.plotting import plot_histogram_with_fit, plot_boxplot, plot_qq
from utils.validation import get_numeric_columns


def render_distribution_page():
    """Render the Distribution Analysis page."""
    st.header("📉 Probability Distribution Analysis")
    st.markdown(
        "Evaluating the empirical shape of a variable is essential before interpreting parametric statistics. "
        "Fitted normal curves are displayed strictly as **approximative estimators**, not proof of a Gaussian data-generating process."
    )

    from ui import check_dataset_and_analysis_ready
    if not check_dataset_and_analysis_ready():
        return

    working_df = st.session_state.get("working_df")

    num_cols = get_numeric_columns(working_df)
    if not num_cols:
        st.error("No numerical columns found in the active dataset.")
        return

    selected_cols = st.session_state.get("selected_columns", [])
    if not selected_cols:
        selected_cols = num_cols

    target_var = st.selectbox(
        "Select Variable for Distributional Analysis",
        options=selected_cols,
        key="dist_var_select"
    )

    if not target_var:
        return

    # Analyze distribution
    series = working_df[target_var]
    dist_info = analyze_distribution(series)

    # Calculate IQR fences for reference
    iqr_res = detect_iqr_outliers(series, multiplier=st.session_state.get("iqr_multiplier", 1.5))

    # Summary metrics cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Symmetry Status", dist_info["symmetry_status"])
    with m2:
        st.metric("Tail Property", dist_info["tail_status"])
    with m3:
        st.metric("Sample Skewness", f"{dist_info['skewness']:.3f}" if pd_notna(dist_info['skewness']) else "N/A")
    with m4:
        st.metric("Excess Kurtosis", f"{dist_info['kurtosis']:.3f}" if pd_notna(dist_info['kurtosis']) else "N/A")

    st.info(f"📋 **Empirical Shape Summary:** {dist_info['shape_summary']}")

    st.divider()

    # Visualizations Tabs
    tab_hist, tab_box, tab_qq = st.tabs(["📊 Histogram & Density Fit", "📦 Tukey Boxplot", "📈 Normal Q-Q Plot"])

    with tab_hist:
        show_normal = st.checkbox(
            "Overlay Estimated Normal PDF Curve",
            value=dist_info["fit_normal_meaningful"],
            help="Estimates a Gaussian density based on sample mean and sample standard deviation."
        )
        fig_hist = plot_histogram_with_fit(
            series=series,
            var_name=target_var,
            iqr_lower=iqr_res["lower_fence"],
            iqr_upper=iqr_res["upper_fence"],
            show_normal_fit=show_normal
        )
        st.pyplot(fig_hist)
        st.caption("Dotted red lines denote Tukey's outer fences (Q1 - 1.5×IQR and Q3 + 1.5×IQR).")

    with tab_box:
        fig_box = plot_boxplot(
            series=series,
            var_name=target_var,
            iqr_multiplier=st.session_state.get("iqr_multiplier", 1.5)
        )
        st.pyplot(fig_box)
        st.caption("Orange vertical line indicates sample median; horizontal whiskers mark Tukey fences.")

    with tab_qq:
        fig_qq = plot_qq(series=series, var_name=target_var)
        st.pyplot(fig_qq)
        st.caption("Points hugging the theoretical line indicate compatibility with a Gaussian distribution; S-curves or upward curvature denote skewness or heavy tails.")


def pd_notna(val):
    import pandas as pd
    return pd.notna(val)
