"""
UI components and Streamlit pages package.
Provides workflow navigation gating and page guard helpers.
"""

import streamlit as st


def check_dataset_and_analysis_ready(allow_pre_analysis_quality: bool = False) -> bool:
    """
    Workflow guard ensuring analysis proceeds only after a valid CSV has been uploaded
    and the user has confirmed analysis by clicking 'Analyze Dataset'.
    """
    working_df = st.session_state.get("working_df", None)
    if working_df is None:
        st.warning("⚠️ **No dataset uploaded yet.** Please upload your CSV dataset to begin statistical analysis.")
        if st.button("📤 Go to Dataset Upload", key="guard_goto_upload"):
            st.session_state["pending_nav"] = "2. Dataset Import"
            st.session_state["current_page"] = "2. Dataset Import"
            st.rerun()
        return False

    if allow_pre_analysis_quality:
        return True

    is_ready = st.session_state.get("analysis_ready", False)
    if not is_ready:
        dataset_name = st.session_state.get("dataset_name", "your file")
        st.info(
            f"ℹ️ **Dataset '{dataset_name}' is uploaded, but analysis has not been launched yet.** "
            "Please go to **1. Upload Dataset**, verify your selected numerical variables, and click **'Analyze Dataset'**."
        )
        if st.button("🚀 Go to Dataset Upload & Click 'Analyze Dataset'", key="guard_goto_analyze"):
            st.session_state["pending_nav"] = "2. Dataset Import"
            st.session_state["current_page"] = "2. Dataset Import"
            st.rerun()
        return False

    return True
