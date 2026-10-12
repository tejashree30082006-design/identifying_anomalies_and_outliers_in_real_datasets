"""
Export and Reporting page (Module L).
Enables one-click generation and downloading of CSV tables, sanitized JSON summaries,
and reproducible Markdown audit reports.
"""

import streamlit as st
import pandas as pd

from core.descriptive_stats import calculate_descriptive_statistics
from core.data_quality import analyze_data_quality
from core.outlier_detection import detect_all_outliers
from core.hypothesis_testing import test_normality, test_grubbs_outlier
from core.result_comparison import compare_detection_methods
from core.report_generator import generate_markdown_report
from utils.export_utils import (
    export_flagged_records_csv,
    export_dataframe_to_csv,
    export_analysis_to_json
)
from utils.validation import get_numeric_columns


def render_export_page():
    """Render the Export and Reporting page."""
    st.header("📄 Export & Reproducible Reporting")
    st.markdown(
        "Generate and download transparent analytical assets for academic evaluation, peer review, "
        "or compliance audits. All exports reflect calculated findings and contain full methodological provenance."
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

    dataset_name = st.session_state.get("dataset_name", "dataset")
    iqr_k = float(st.session_state.get("iqr_multiplier", 1.5))
    z_thresh = float(st.session_state.get("z_threshold", 3.0))
    alpha_val = float(st.session_state.get("alpha", 0.05))

    # Compile comprehensive analytical bundle
    with st.spinner("Compiling analysis results..."):
        quality = analyze_data_quality(working_df)
        desc_stats = calculate_descriptive_statistics(working_df, selected_cols)
        detection_bundle = detect_all_outliers(working_df, selected_cols, z_threshold=z_thresh, iqr_multiplier=iqr_k)
        comp_results = compare_detection_methods(detection_bundle["combined_df"], detection_bundle["combined_df"])
        comp_df = comp_results["comparison_df"]

        norm_tests = {col: test_normality(working_df[col], alpha=alpha_val) for col in selected_cols}
        grubbs_tests = {col: test_grubbs_outlier(working_df[col], alpha=alpha_val) for col in selected_cols}

        analysis_bundle = {
            "dataset_name": dataset_name,
            "selected_columns": selected_cols,
            "iqr_multiplier": iqr_k,
            "z_threshold": z_thresh,
            "alpha": alpha_val,
            "quality": quality,
            "desc_stats": desc_stats,
            "outlier_summary": comp_results,
            "normality_tests": norm_tests,
            "grubbs_tests": grubbs_tests
        }

        # Generate markdown report
        md_report = generate_markdown_report(analysis_bundle)

    # --- Section 1: Downloads ---
    st.subheader("1. Download Analytical Deliverables")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("##### 📊 Tabular Data (CSV)")
        # 1. Flagged records CSV
        csv_flagged = export_flagged_records_csv(comp_df)
        st.download_button(
            label="📥 Download Flagged Records (CSV)",
            data=csv_flagged,
            file_name=f"{dataset_name}_flagged_records.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2. Descriptive statistics CSV
        csv_stats = export_dataframe_to_csv(desc_stats)
        st.download_button(
            label="📥 Download Descriptive Statistics (CSV)",
            data=csv_stats,
            file_name=f"{dataset_name}_descriptive_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 3. All detection results CSV
        csv_all = export_dataframe_to_csv(comp_df)
        st.download_button(
            label="📥 Download All Detection Results (CSV)",
            data=csv_all,
            file_name=f"{dataset_name}_full_detection_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d_col2:
        st.markdown("##### 📝 Structured Reports (Markdown & JSON)")
        # 4. Markdown Report
        st.download_button(
            label="📥 Download Full Audit Report (Markdown .md)",
            data=md_report,
            file_name=f"{dataset_name}_audit_report.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary"
        )

        # 5. JSON Summary
        json_str = export_analysis_to_json(analysis_bundle)
        st.download_button(
            label="📥 Download Structured Analysis Summary (JSON)",
            data=json_str,
            file_name=f"{dataset_name}_analysis_summary.json",
            mime="application/json",
            use_container_width=True
        )

    st.divider()

    # --- Section 2: Report Preview ---
    st.subheader("2. Report Preview")
    with st.expander("Inspect Generated Markdown Audit Report", expanded=True):
        st.markdown(md_report)
