"""
Descriptive Statistics page (Module C).
Computes summary metrics, central tendency, dispersion, quartiles, skewness, and kurtosis.
Presents findings in formatted tables accompanied by plain-English interpretations.
"""

import streamlit as st
import pandas as pd
import numpy as np

from core.descriptive_stats import calculate_descriptive_statistics
from utils.validation import get_numeric_columns


def render_descriptive_stats_page():
    """Render the Descriptive Statistics page."""
    st.header("📈 Descriptive Statistics & Summary Metrics")
    st.markdown(
        "Descriptive statistics summarize sample distributions without presupposing normality. "
        "Sample variance and standard deviation are computed using unbiased sample estimators ($ddof=1$). "
        "Kurtosis reflects Fisher's excess kurtosis (where a standard normal distribution equals 0.0)."
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

    # Compute descriptive statistics
    stats_df = calculate_descriptive_statistics(working_df, selected_cols)
    if stats_df.empty:
        st.info("No statistics calculated. Please select at least one numerical variable.")
        return

    st.subheader("1. Summary Metrics Table")
    st.caption("Rows display statistical parameters; columns represent selected variables.")

    # Formatted display DataFrame
    display_df = stats_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating, np.integer)) and pd.notna(v) and not np.isinf(v) else str(v)
        )
    st.dataframe(display_df, use_container_width=True)

    st.divider()

    # Plain-English Variable Interpretations
    st.subheader("2. Plain-English Statistical Interpretations")
    for col in selected_cols:
        if col not in stats_df.columns:
            continue

        c_mean = stats_df.loc["Mean", col]
        c_median = stats_df.loc["Median", col]
        c_std = stats_df.loc["Sample Std Dev (s, ddof=1)", col]
        c_iqr = stats_df.loc["IQR (Q3 - Q1)", col]
        c_skew = stats_df.loc["Skewness (Sample)", col]
        c_kurt = stats_df.loc["Kurtosis (Fisher's Excess)", col]
        c_min = stats_df.loc["Minimum", col]
        c_max = stats_df.loc["Maximum", col]
        c_status = stats_df.loc["Status / Notes", col]

        with st.expander(f"📌 Variable Interpretation: {col}", expanded=True):
            if c_status != "Normal":
                st.warning(f"Special Status: **{c_status}**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"- **Center:** Mean = `{c_mean:.2f}`, Median = `{c_median:.2f}`")
                st.markdown(f"- **Dispersion:** Sample Std Dev ($s$) = `{c_std:.2f}`, IQR = `{c_iqr:.2f}`")
                st.markdown(f"- **Range:** `{c_min:.2f}` to `{c_max:.2f}` (Spread = `{c_max - c_min:.2f}`)")

            with col_b:
                skew_desc = "Symmetric" if abs(c_skew) < 0.5 else ("Right-skewed (positive)" if c_skew > 0 else "Left-skewed (negative)")
                kurt_desc = "Mesokurtic (normal tails)" if abs(c_kurt) < 0.5 else ("Leptokurtic (heavy-tailed)" if c_kurt > 0 else "Platykurtic (light-tailed)")
                st.markdown(f"- **Skewness:** `{c_skew:.3f}` ({skew_desc})")
                st.markdown(f"- **Excess Kurtosis:** `{c_kurt:.3f}` ({kurt_desc})")

            # Narrative conclusion
            if abs(c_mean - c_median) > (0.25 * c_std) if pd.notna(c_std) and c_std > 0 else False:
                st.markdown(
                    f"💡 *The divergence between the mean ({c_mean:.2f}) and median ({c_median:.2f}) indicates that extreme values "
                    f"are pulling the parametric mean toward the tail. The median and IQR provide more resistant measures of location and scale.*"
                )
            else:
                st.markdown(
                    f"💡 *Mean and median are closely aligned, suggesting relative balance in the central 50% of the distribution.*"
                )
