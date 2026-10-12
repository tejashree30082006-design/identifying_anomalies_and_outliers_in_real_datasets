"""
Clean, minimal CSV Dataset Upload page (Module B).
Focuses strictly on user-provided CSV upload, validation, inspection, and column selection.
Avoids infinite rerun loops and redundant parsing.
"""

import streamlit as st
import pandas as pd

from core.data_loader import load_csv, load_demo_dataset, prepare_dataset
from utils.validation import get_numeric_columns


def render_dataset_page():
    """Render the focused, minimal CSV upload and column selection page."""
    st.header("Upload Your Dataset")
    st.markdown("Upload a CSV file to begin statistical analysis.")

    # 1. Streamlit CSV file uploader
    uploaded_file = st.file_uploader(
        "Upload a CSV file to begin statistical analysis.",
        type=["csv"],
        help="Upload standard comma-separated values (.csv) file up to 50 MB.",
        key="main_csv_uploader",
        label_visibility="collapsed"
    )

    # 2. Detect genuinely new upload or file removal (cached per file identifier)
    if uploaded_file is not None:
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("loaded_file_id") != file_identifier:
            try:
                raw_df = load_csv(uploaded_file)
                working_df = prepare_dataset(raw_df)
                num_cols = get_numeric_columns(working_df)

                st.session_state["raw_df"] = raw_df
                st.session_state["working_df"] = working_df
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["loaded_file_id"] = file_identifier
                st.session_state["analysis_ready"] = False
                st.session_state["selected_columns"] = num_cols
                st.session_state["upload_error"] = None
                st.session_state["is_demo"] = False
            except Exception as e:
                st.session_state["raw_df"] = None
                st.session_state["working_df"] = None
                st.session_state["dataset_name"] = None
                st.session_state["loaded_file_id"] = file_identifier
                st.session_state["analysis_ready"] = False
                st.session_state["selected_columns"] = []
                st.session_state["upload_error"] = str(e)
    else:
        # If user removed the file from the uploader
        if st.session_state.get("loaded_file_id") is not None and not st.session_state.get("is_demo", False):
            st.session_state["raw_df"] = None
            st.session_state["working_df"] = None
            st.session_state["dataset_name"] = None
            st.session_state["loaded_file_id"] = None
            st.session_state["analysis_ready"] = False
            st.session_state["selected_columns"] = []
            st.session_state["upload_error"] = None

    # Optional demonstration dataset (remains strictly optional and secondary)
    with st.expander("🧪 (Optional) Alternatively, explore with the built-in demo dataset", expanded=False):
        st.caption("Load synthetic retail transaction dataset (71 rows) for testing.")
        if st.button("Load Built-in Demo Dataset", key="btn_load_demo_optional"):
            try:
                raw_df = load_demo_dataset()
                working_df = prepare_dataset(raw_df)
                num_cols = get_numeric_columns(working_df)
                st.session_state["raw_df"] = raw_df
                st.session_state["working_df"] = working_df
                st.session_state["dataset_name"] = "Synthetic Demo Dataset"
                st.session_state["loaded_file_id"] = "builtin_demo"
                st.session_state["is_demo"] = True
                st.session_state["analysis_ready"] = False
                st.session_state["selected_columns"] = num_cols
                st.session_state["upload_error"] = None
                st.rerun()
            except Exception as e:
                st.error(f"Error loading demo dataset: {str(e)}")

    # 3. Validation error display
    if st.session_state.get("upload_error"):
        st.error(f"Error processing CSV file: {st.session_state['upload_error']}")
        return

    # Before a file is uploaded, stop here!
    working_df = st.session_state.get("working_df", None)
    if working_df is None:
        return

    # 4. After a valid file is uploaded:
    dataset_name = st.session_state.get("dataset_name", "Uploaded Dataset")
    st.success(f"Successfully loaded '{dataset_name}'.")

    # Dimensions
    user_cols = [c for c in working_df.columns if not str(c).startswith("_")]
    col_rows, col_cols = st.columns(2)
    with col_rows:
        st.metric("Number of Rows", f"{len(working_df):,}")
    with col_cols:
        st.metric("Number of Columns", f"{len(user_cols):,}")

    # Dataset Preview
    st.markdown("##### Dataset Preview")
    st.dataframe(working_df[user_cols].head(8), use_container_width=True)

    # Numerical Columns Validation
    num_cols = get_numeric_columns(working_df)
    if not num_cols:
        st.error(
            "⚠️ The uploaded dataset contains no numerical columns. "
            "Numerical columns are required for statistical analysis. "
            "Please upload a dataset containing numerical variables."
        )
        return

    # Column Selection
    current_selected = st.session_state.get("selected_columns", num_cols)
    valid_defaults = [c for c in current_selected if c in num_cols] or num_cols

    chosen_cols = st.multiselect(
        "Available Numerical Columns (select one or multiple to analyze):",
        options=num_cols,
        default=valid_defaults,
        key="selected_num_columns_multiselect"
    )
    st.session_state["selected_columns"] = chosen_cols

    # Analyze Dataset Button
    st.markdown("")
    btn_c1, btn_c2, btn_c3 = st.columns([1, 2, 1])
    with btn_c2:
        if st.button("🚀 Analyze Dataset", type="primary", use_container_width=True, key="btn_run_analysis_clean"):
            if not chosen_cols:
                st.warning("Please select at least one numerical column to analyze.")
            else:
                st.session_state["analysis_ready"] = True
                st.session_state["selected_columns"] = chosen_cols
                # Request navigation to next page via pending_nav (avoids widget-key conflict)
                st.session_state["pending_nav"] = "3. Data Quality Audit"
                st.session_state["current_page"] = "3. Data Quality Audit"
                st.rerun()

    if st.session_state.get("analysis_ready", False):
        st.success(f"Analysis ready for {len(chosen_cols)} variable(s): `{', '.join(chosen_cols)}`. Navigate in the sidebar to view statistics and charts.")
