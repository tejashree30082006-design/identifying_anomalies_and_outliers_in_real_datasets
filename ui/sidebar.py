"""
Sidebar navigation and configuration panel.
Provides unified routing, dataset status monitoring, and analytical parameter controls.
"""

import streamlit as st
from config.settings import (
    APP_TITLE,
    VERSION,
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_Z_THRESHOLD,
    DEFAULT_ALPHA
)
from utils.validation import get_numeric_columns


def render_sidebar():
    """Render the application sidebar with navigation and parameter controls."""
    with st.sidebar:
        st.markdown(f"### 📊 {APP_TITLE}")
        st.caption(f"Academic Evaluation Edition • v{VERSION}")
        st.divider()

        # Primary Navigation
        pages = [
            "1. Home & Overview",
            "2. Dataset Import",
            "3. Data Quality Audit",
            "4. Descriptive Statistics",
            "5. Distribution Analysis",
            "6. Outlier Detection (IQR & Z-Score)",
            "7. Hypothesis Testing",
            "8. Results & Comparison",
            "9. Export & Reporting"
        ]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = pages[0]

        # Consume any pending navigation request BEFORE the radio is instantiated
        pending = st.session_state.pop("pending_nav", None)
        if pending and pending in pages:
            st.session_state["current_page"] = pending

        selected_page = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0,
            key="nav_radio"
        )
        st.session_state["current_page"] = selected_page

        st.divider()

        # Dataset Status Widget
        st.markdown("#### 📁 Active Dataset")
        working_df = st.session_state.get("working_df", None)
        dataset_name = st.session_state.get("dataset_name", "None loaded")
        is_demo = st.session_state.get("is_demo", False)

        if working_df is not None:
            tag = " [Demo Synthetic]" if is_demo else " [User Upload]"
            is_ready = st.session_state.get("analysis_ready", False)
            if is_ready:
                st.success(f"**{dataset_name}**{tag}\n\n*Status: Analyzed*")
            else:
                st.warning(f"**{dataset_name}**{tag}\n\n*Status: Pending Analysis*")

            st.markdown(f"- **Rows:** {len(working_df):,}")
            st.markdown(f"- **Columns:** {len(working_df.columns):,}")
            
            num_cols = get_numeric_columns(working_df)
            st.markdown(f"- **Numeric Variables:** {len(num_cols)}")

            # Variable selection
            if "selected_columns" not in st.session_state or not st.session_state["selected_columns"]:
                st.session_state["selected_columns"] = num_cols

            selected_cols = st.multiselect(
                "Analyzed Variables",
                options=num_cols,
                default=[c for c in st.session_state["selected_columns"] if c in num_cols],
                key="sidebar_col_select"
            )
            st.session_state["selected_columns"] = selected_cols

            if st.button("🔄 Upload Different CSV / Reset", use_container_width=True):
                st.session_state["working_df"] = None
                st.session_state["raw_df"] = None
                st.session_state["dataset_name"] = "None loaded"
                st.session_state["analysis_ready"] = False
                st.session_state["selected_columns"] = []
                st.session_state["pending_nav"] = "2. Dataset Import"
                st.session_state["current_page"] = "2. Dataset Import"
                st.rerun()
        else:
            st.info("No active dataset. Upload a CSV under **2. Dataset Import** to begin.")

        st.divider()

        # Statistical Thresholds Configuration
        st.markdown("#### ⚙️ Statistical Parameters")
        with st.expander("Configure Thresholds", expanded=False):
            iqr_mult = st.slider(
                "Tukey IQR Multiplier (k)",
                min_value=1.0,
                max_value=3.0,
                value=float(st.session_state.get("iqr_multiplier", DEFAULT_IQR_MULTIPLIER)),
                step=0.1,
                help="Standard Tukey fence multiplier (default 1.5)."
            )
            st.session_state["iqr_multiplier"] = iqr_mult

            z_thresh = st.slider(
                "Z-Score Absolute Threshold (|Z|)",
                min_value=1.5,
                max_value=5.0,
                value=float(st.session_state.get("z_threshold", DEFAULT_Z_THRESHOLD)),
                step=0.25,
                help="Standard threshold for parametric outlier detection (default 3.0 standard deviations)."
            )
            st.session_state["z_threshold"] = z_thresh

            alpha_val = st.select_slider(
                "Significance Level (α)",
                options=[0.01, 0.05, 0.10],
                value=float(st.session_state.get("alpha", DEFAULT_ALPHA)),
                help="Type I error rate threshold for hypothesis tests (default 0.05)."
            )
            st.session_state["alpha"] = alpha_val

        return selected_page
