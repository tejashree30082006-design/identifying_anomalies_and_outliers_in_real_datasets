"""
Main Streamlit application entry point.
Initializes session state, applies clean academic styles, and routes between analytical modules.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path to ensure module imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import APP_TITLE, VERSION
from ui.sidebar import render_sidebar
from ui.home import render_home_page
from ui.dataset_page import render_dataset_page
from ui.data_quality_page import render_data_quality_page
from ui.descriptive_stats_page import render_descriptive_stats_page
from ui.distribution_page import render_distribution_page
from ui.outlier_page import render_outlier_page
from ui.hypothesis_page import render_hypothesis_page
from ui.results_page import render_results_page
from ui.export_page import render_export_page


def apply_custom_styles():
    """Apply modern, clean, academic CSS styling."""
    st.markdown(
        """
        <style>
        /* General page typography and card styles */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        h1, h2, h3, h4 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1a252f;
            letter-spacing: -0.02em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2b5c8f;
        }
        .stAlert {
            border-radius: 8px;
        }
        /* Buttons and interactive tabs */
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
        }
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def initialize_session_state():
    """Initialize persistent Streamlit session state defaults."""
    defaults = {
        "raw_df": None,
        "working_df": None,
        "dataset_name": "None loaded",
        "is_demo": False,
        "selected_columns": [],
        "iqr_multiplier": 1.5,
        "z_threshold": 3.0,
        "alpha": 0.05,
        "current_page": "1. Home & Overview"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    st.set_page_config(
        page_title=f"{APP_TITLE} (v{VERSION})",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_styles()
    initialize_session_state()

    # Render sidebar navigation
    selected_page = render_sidebar()

    # Route to selected module
    if "1. Home" in selected_page:
        render_home_page()
    elif "2. Dataset Import" in selected_page:
        render_dataset_page()
    elif "3. Data Quality" in selected_page:
        render_data_quality_page()
    elif "4. Descriptive Statistics" in selected_page:
        render_descriptive_stats_page()
    elif "5. Distribution Analysis" in selected_page:
        render_distribution_page()
    elif "6. Outlier Detection" in selected_page:
        render_outlier_page()
    elif "7. Hypothesis Testing" in selected_page:
        render_hypothesis_page()
    elif "8. Results & Comparison" in selected_page:
        render_results_page()
    elif "9. Export & Reporting" in selected_page:
        render_export_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
