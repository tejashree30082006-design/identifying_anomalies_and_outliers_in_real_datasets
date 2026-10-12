"""
Data Quality Audit page (Module B).
Evaluates missing values, duplicate records, non-finite values, and constant columns
with transparent visualizations and handling policies.
"""

import streamlit as st
import pandas as pd

from core.data_quality import analyze_data_quality
from utils.plotting import plot_missing_values


def render_data_quality_page():
    """Render the Data Quality Audit page."""
    st.header("🧹 Data Quality & Integrity Audit")
    st.markdown(
        "A rigorous statistical analysis requires examining data hygiene prior to modeling. "
        "Missing entries, duplicate rows, infinite values, and zero-variance columns must be cataloged "
        "without silently modifying or deleting the underlying source data."
    )

    from ui import check_dataset_and_analysis_ready
    if not check_dataset_and_analysis_ready(allow_pre_analysis_quality=True):
        return

    working_df = st.session_state.get("working_df")

    quality = analyze_data_quality(working_df)

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Missing Cells", f"{quality['total_missing_cells']:,}")
    with c2:
        dup_cnt = quality["duplicate_summary"]["duplicate_count"]
        st.metric("Duplicate Rows", f"{dup_cnt:,} ({quality['duplicate_summary']['duplicate_percentage']}%)")
    with c3:
        inf_cnt = sum(quality["infinite_counts"].values())
        st.metric("Infinite / Non-Finite", f"{inf_cnt:,}")
    with c4:
        const_cnt = len(quality["constant_columns"])
        st.metric("Constant Columns", f"{const_cnt:,}")

    st.divider()

    # Section 1: Missing Values
    st.subheader("1. Missing Values Analysis")
    missing_df = quality["missing_summary"]
    if not missing_df.empty:
        col_table, col_chart = st.columns([1, 1], gap="medium")
        with col_table:
            st.markdown("##### Missingness Table by Column")
            st.dataframe(missing_df, use_container_width=True)
            
            has_missing = quality["total_missing_cells"] > 0
            if has_missing:
                st.info(
                    "📌 **Handling Policy:** Missing values (`NaN`) are preserved in the original dataset "
                    "and explicitly excluded during numerical point calculations (complete-case policy). "
                    "Flagged outliers are never imputed or deleted."
                )
            else:
                st.success("✅ Complete data coverage: zero missing values detected across all columns.")

        with col_chart:
            fig_miss = plot_missing_values(missing_df)
            st.pyplot(fig_miss)

    st.divider()

    # Section 2: Duplicate Records Audit
    st.subheader("2. Duplicate Records Audit")
    dup_info = quality["duplicate_summary"]
    if dup_info["has_duplicates"]:
        st.warning(
            f"⚠️ Detected **{dup_info['duplicate_count']}** duplicate row(s) "
            f"({dup_info['duplicate_percentage']}% of dataset). "
            "Internal record identifiers ensure each duplicate row is independently tracked."
        )
        # Show duplicate rows
        user_cols = [c for c in working_df.columns if not str(c).startswith("_")]
        dup_rows = working_df[working_df.duplicated(subset=user_cols, keep=False)][user_cols]
        with st.expander("Inspect Duplicate Records", expanded=True):
            st.dataframe(dup_rows, use_container_width=True)
    else:
        st.success("✅ No duplicate records detected in the dataset.")

    st.divider()

    # Section 3: Non-Finite & Constant Columns
    st.subheader("3. Numeric Anomalies & Edge Cases")
    col_inf, col_const = st.columns(2)

    with col_inf:
        st.markdown("##### Infinite Values (`+inf`, `-inf`)")
        if quality["infinite_counts"]:
            st.error(f"Infinite values detected: {quality['infinite_counts']}")
            st.caption("Infinite observations are excluded from variance and Z-score calculations.")
        else:
            st.success("✅ Zero infinite values detected.")

    with col_const:
        st.markdown("##### Constant Columns (Zero Variance)")
        if quality["constant_columns"]:
            st.warning(f"Constant columns detected: `{', '.join(quality['constant_columns'])}`")
            st.caption("Zero-variance columns produce undefined Z-scores and zero-IQR fences.")
        else:
            st.success("✅ All numerical variables exhibit non-zero variance.")
