"""
Outlier Detection page (Modules E and F).
Provides independent Tukey Interquartile Range (IQR) and Z-score outlier detection,
displaying mathematical parameters, deviation tables, and specialized diagnostic plots.
"""

import streamlit as st
import pandas as pd
import numpy as np

from core.outlier_detection import detect_iqr_outliers, detect_zscore_outliers
from utils.plotting import plot_boxplot, plot_zscores
from utils.validation import get_numeric_columns


def render_outlier_page():
    """Render the Outlier Detection page for IQR and Z-Score methods."""
    st.header("🎯 Outlier Detection: IQR & Z-Score Methods")
    st.markdown(
        "Compare non-parametric resistant fences (Tukey's IQR rule) against parametric dispersion thresholds (Z-score). "
        "Observations are flagged relative to each method's mathematical definition without assuming malicious data fabrication."
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
        "Select Variable for Outlier Inspection",
        options=selected_cols,
        key="outlier_var_select"
    )

    if not target_var:
        return

    series = working_df[target_var]
    iqr_k = float(st.session_state.get("iqr_multiplier", 1.5))
    z_thresh = float(st.session_state.get("z_threshold", 3.0))

    tab_iqr, tab_z = st.tabs(["📦 Method 1: Tukey IQR Fences", "📏 Method 2: Standard Z-Score"])

    # --- TAB 1: IQR Method ---
    with tab_iqr:
        iqr_res = detect_iqr_outliers(series, multiplier=iqr_k)

        st.subheader("Tukey Interquartile Range (IQR) Rule")
        st.markdown(
            f"""
            - **First Quartile (Q1):** `{iqr_res['q1']:.2f}`
            - **Third Quartile (Q3):** `{iqr_res['q3']:.2f}`
            - **Interquartile Range (IQR):** `{iqr_res['iqr']:.2f}`
            - **Lower Fence ($Q_1 - {iqr_k} \\times \\text{{IQR}}$):** `{iqr_res['lower_fence']:.2f}`
            - **Upper Fence ($Q_3 + {iqr_k} \\times \\text{{IQR}}$):** `{iqr_res['upper_fence']:.2f}`
            - **Total Observations Flagged:** `{iqr_res['flagged_count']}` / `{iqr_res['valid_count']}`
            """
        )

        if iqr_res["iqr_is_zero"]:
            st.error(iqr_res["notes"])

        # Flagged table
        iqr_sub = iqr_res["results_df"]
        flagged_iqr = iqr_sub[iqr_sub["iqr_flag"] == True] if not iqr_sub.empty else pd.DataFrame()

        if not flagged_iqr.empty:
            st.markdown(f"##### Flagged Observations Table ({len(flagged_iqr)} records)")
            cols_to_show = ["_original_row_number", "original_value", "iqr_status", "iqr_deviation"]
            st.dataframe(flagged_iqr[cols_to_show], use_container_width=True)
        else:
            st.success(f"✅ No observations fall outside the Tukey fences (k={iqr_k}) for '{target_var}'.")

        # Visualization
        flagged_vals = flagged_iqr["numeric_value"].tolist() if not flagged_iqr.empty else []
        fig_box = plot_boxplot(series, target_var, iqr_multiplier=iqr_k, flagged_values=flagged_vals)
        st.pyplot(fig_box)

    # --- TAB 2: Z-Score Method ---
    with tab_z:
        z_res = detect_zscore_outliers(series, threshold=z_thresh)

        st.subheader("Standard Z-Score Method")
        if not z_res["is_defined"]:
            st.warning(f"⚠️ {z_res['notes']}")
        else:
            st.markdown(
                f"""
                - **Sample Mean ($\\bar{{x}}$):** `{z_res['mean']:.2f}`
                - **Sample Standard Deviation ($s$, $ddof=1$):** `{z_res['std']:.2f}`
                - **Configured Threshold (|Z|):** `>{z_thresh:.1f}σ`
                - **Flagged Observations:** `{z_res['flagged_count']}` / `{z_res['valid_count']}`
                """
            )

            st.caption(
                "📌 **Methodological Assumption:** Z-scores are most interpretable when data is approximately normal. "
                "Skewness or extreme values can inflate the sample standard deviation, dampening Z-scores for other outliers (masking effect)."
            )

            z_sub = z_res["results_df"]
            flagged_z = z_sub[z_sub["zscore_flag"] == True] if not z_sub.empty else pd.DataFrame()

            if not flagged_z.empty:
                st.markdown(f"##### Flagged Observations Table ({len(flagged_z)} records)")
                cols_to_show = ["_original_row_number", "original_value", "signed_zscore", "zscore_direction", "zscore_excess"]
                st.dataframe(flagged_z[cols_to_show], use_container_width=True)
            else:
                st.success(f"✅ No observations exceed the |Z| > {z_thresh:.1f} threshold for '{target_var}'.")

            # Visualization
            fig_z = plot_zscores(
                series=series,
                var_name=target_var,
                threshold=z_thresh,
                zscore_series=z_sub["signed_zscore"] if not z_sub.empty else None
            )
            st.pyplot(fig_z)
