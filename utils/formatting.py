"""
Formatting and presentation helper utilities.
Handles clean numeric display, p-value scientific notation, and metric formatting.
"""

from typing import Union
import math
import numpy as np
import pandas as pd


def format_number(val: Union[int, float, None], decimals: int = 2) -> str:
    """Format numeric values safely with specified decimal precision."""
    if val is None or pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, (float, np.floating)):
        if np.isinf(val):
            return "∞" if val > 0 else "-∞"
        return f"{val:.{decimals}f}"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return str(val)


def format_p_value(p: Union[float, None]) -> str:
    """Format statistical p-value appropriately (scientific notation for small values)."""
    if p is None or pd.isna(p) or math.isnan(p):
        return "N/A"
    if p < 0.0001:
        return f"{p:.4e}"
    return f"{p:.4f}"


def format_percentage(val: Union[float, int, None]) -> str:
    """Format ratio or percentage to readable string."""
    if val is None or pd.isna(val) or math.isnan(val):
        return "0.0%"
    return f"{val:.1f}%"
