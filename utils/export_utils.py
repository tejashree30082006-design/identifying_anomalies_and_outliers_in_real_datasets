"""
Export utilities module.
Generates CSV tables, JSON-safe metadata summaries, and downloadable reports
with strict sanitization of non-finite floating point numbers and NumPy data types.
"""

from typing import Dict, Any, Union
import json
import numpy as np
import pandas as pd


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize objects to standard JSON-safe types.
    Converts NaN to None, infinities to strings/None, and numpy scalars to Python native types.
    """
    if isinstance(obj, (pd.DataFrame,)):
        return sanitize_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, (pd.Series,)):
        return sanitize_for_json(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def export_dataframe_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to standard CSV string."""
    if df is None or df.empty:
        return ""
    return df.to_csv(index=True if df.index.name else False, encoding="utf-8")


def export_flagged_records_csv(comparison_df: pd.DataFrame) -> str:
    """Filter comparison DataFrame to include only flagged observations and export as CSV."""
    if comparison_df is None or comparison_df.empty:
        return ""
    flagged_mask = (comparison_df["iqr_flag"] == True) | (comparison_df["zscore_flag"] == True)
    flagged_df = comparison_df[flagged_mask].copy()
    return flagged_df.to_csv(index=False, encoding="utf-8")


def export_analysis_to_json(analysis_results: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert comprehensive analysis results dictionary to a sanitized, JSON-compliant string.
    """
    safe_data = sanitize_for_json(analysis_results)
    return json.dumps(safe_data, indent=indent, ensure_ascii=False)
