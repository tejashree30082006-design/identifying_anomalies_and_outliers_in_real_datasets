"""
Home and entrypoint wrapper.
Directs directly to the clean, minimal Upload Your Dataset interface.
"""

from ui.dataset_page import render_dataset_page


def render_home_page():
    """Render the clean upload interface."""
    render_dataset_page()
