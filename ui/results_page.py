"""
Results, Comparison, and Suspicious-Record Investigation page (Modules G and J).
Synthesizes Tukey IQR and Z-Score detection into four mutually exclusive categories.
Provides drill-down investigation of source records and method concordance analysis.
"""

import streamlit as st
import pandas as pd

from core.outlier_detection import detect_all_outliers
from core.result_comparison import compare_detection_methods
from utils.plotting import plot_method_comparison_counts
from utils.validation import get_numeric_columns
from config.settings import (
    CATEGORY_BOTH,
    CATEGORY_IQR_ONLY,
    CATEGORY_ZSCORE_ONLY,
    CATEGORY_NEITHER
)


def render_results_page():
    """Render the Results Comparison and Investigation page."""
    st.header("🔍 Results Synthesis & Suspicious-Record Investigation")
    st.markdown(
        "Synthesize results across both detection methodologies into **four mutually exclusive categories** "
        "to avoid double-counting. Trace flagged observations back to their original source rows."
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

    z_thresh = float(st.session_state.get("z_threshold", 3.0))
    iqr_k = float(st.session_state.get("iqr_multiplier", 1.5))

    # Run combined detection
    detection_bundle = detect_all_outliers(
        df=working_df,
        selected_columns=selected_cols,
        z_threshold=z_thresh,
        iqr_multiplier=iqr_k
    )

    comp_results = compare_detection_methods(
        iqr_results=detection_bundle["combined_df"],
        zscore_results=detection_bundle["combined_df"]
    )

    comp_df = comp_results["comparison_df"]
    cat_counts = comp_results["category_counts"]
    unique_summary = comp_results["unique_records_summary"]

    # --- Section 1: Overview Cards ---
    st.subheader("1. Mutually Exclusive Method Concordance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔴 Both Methods", f"{cat_counts[CATEGORY_BOTH]:,}")
    with c2:
        st.metric("🟠 IQR Only", f"{cat_counts[CATEGORY_IQR_ONLY]:,}")
    with c3:
        st.metric("🟣 Z-Score Only", f"{cat_counts[CATEGORY_ZSCORE_ONLY]:,}")
    with c4:
        st.metric("🟢 Neither Method", f"{cat_counts[CATEGORY_NEITHER]:,}")

    col_chart, col_uniq = st.columns([1, 1], gap="medium")
    with col_chart:
        fig_bar = plot_method_comparison_counts(cat_counts)
        st.pyplot(fig_bar)

    with col_uniq:
        st.markdown("##### Source Records Summary (Deduplicated)")
        st.markdown(
            f"""
            - **Total Unique Source Records:** `{unique_summary['total_unique_records']:,}`
            - **Flagged Unique Records (Any Variable):** `{unique_summary['flagged_unique_records']:,}` ({unique_summary['flagged_record_percentage']}%)
            - **Completely Unflagged Unique Records:** `{unique_summary['unflagged_unique_records']:,}`
            - **Method Agreement Rate:** `{comp_results['method_agreement_pct']}%`
            """
        )
        st.info(
            "📌 **Analytical Insight:** Points flagged by *Both Methods* represent highest statistical priority. "
            "Points flagged by *IQR Only* or *Z-Score Only* highlight discrepancies between resistant medians and parametric means."
        )

    st.divider()

    # --- Section 2: Suspicious-Record Investigation Table ---
    st.subheader("2. Flagged Records Investigation Table")
    st.markdown(
        "Inspect each flagged observation with its original row number, value, signed Z-score, "
        "Tukey fence status, and neutral rule-based classification."
    )

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cat_filter = st.multiselect(
            "Filter by Category",
            options=[CATEGORY_BOTH, CATEGORY_IQR_ONLY, CATEGORY_ZSCORE_ONLY, CATEGORY_NEITHER],
            default=[CATEGORY_BOTH, CATEGORY_IQR_ONLY, CATEGORY_ZSCORE_ONLY]
        )
    with f_col2:
        var_filter = st.multiselect(
            "Filter by Variable",
            options=selected_cols,
            default=selected_cols
        )
    with f_col3:
        show_all = st.checkbox("Include Unflagged Records", value=False)

    filtered_df = comp_df.copy()
    if not show_all and not cat_filter:
        filtered_df = filtered_df[filtered_df["combined_category"] != CATEGORY_NEITHER]
    elif cat_filter:
        filtered_df = filtered_df[filtered_df["combined_category"].isin(cat_filter)]

    if var_filter:
        filtered_df = filtered_df[filtered_df["variable"].isin(var_filter)]

    # Display columns
    display_cols = [
        "_original_row_number",
        "variable",
        "original_value",
        "combined_category",
        "iqr_status",
        "signed_zscore",
        "explanation"
    ]

    st.markdown(f"**Showing {len(filtered_df)} records**")
    st.dataframe(
        filtered_df[display_cols].rename(columns={"_original_row_number": "Row #"}),
        use_container_width=True
    )

    st.divider()

    # --- Section 3: Deep-Dive Source Row Inspection ---
    st.subheader("3. Inspect Full Source Record Context")
    if not filtered_df.empty:
        available_rows = sorted(filtered_df["_original_row_number"].unique())
        selected_row_num = st.selectbox(
            "Select Original Row Number to View Entire Source Record",
            options=available_rows
        )

        match_row = working_df[working_df["_original_row_number"] == selected_row_num]
        if not match_row.empty:
            st.markdown(f"##### Full Source Record (Row #{selected_row_num})")
            user_cols = [c for c in match_row.columns if not str(c).startswith("_")]
            st.dataframe(match_row[user_cols], use_container_width=True)
    else:
        st.info("No records match the current filter selection.")
