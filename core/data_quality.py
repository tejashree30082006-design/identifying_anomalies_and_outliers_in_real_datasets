"""
Data quality audit module.
Evaluates missing values, duplicate rows, infinite values, constant columns,
and column data types without modifying source records.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config.settings import RECORD_ID_COL, ORIGINAL_ROW_COL


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing value counts and percentages for each column.
    Excludes internal metadata columns.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing Percentage", "Valid Count"])

    analysis_cols = [c for c in df.columns if c not in (RECORD_ID_COL, ORIGINAL_ROW_COL)]
    total_rows = len(df)

    rows = []
    for col in analysis_cols:
        missing_count = int(df[col].isna().sum())
        pct = (missing_count / total_rows * 100.0) if total_rows > 0 else 0.0
        valid_count = total_rows - missing_count
        rows.append({
            "Column": str(col),
            "Missing Count": missing_count,
            "Missing Percentage": round(pct, 2),
            "Valid Count": valid_count
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


def get_duplicate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify duplicate rows based on user-provided columns (excluding internal row IDs).
    Returns total count, percentage, and duplicate indices.
    """
    if df is None or df.empty:
        return {
            "duplicate_count": 0,
            "duplicate_percentage": 0.0,
            "has_duplicates": False,
            "duplicate_indices": []
        }

    analysis_cols = [c for c in df.columns if c not in (RECORD_ID_COL, ORIGINAL_ROW_COL)]
    if not analysis_cols:
        return {
            "duplicate_count": 0,
            "duplicate_percentage": 0.0,
            "has_duplicates": False,
            "duplicate_indices": []
        }

    dup_mask = df.duplicated(subset=analysis_cols, keep="first")
    dup_count = int(dup_mask.sum())
    total_rows = len(df)
    dup_pct = (dup_count / total_rows * 100.0) if total_rows > 0 else 0.0
    dup_indices = df.index[dup_mask].tolist()

    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": round(dup_pct, 2),
        "has_duplicates": dup_count > 0,
        "duplicate_indices": dup_indices
    }


def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Conduct an exhaustive data quality audit across the dataset.
    Returns structured results for reporting and UI presentation.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "numeric_columns": [],
            "categorical_columns": [],
            "missing_summary": pd.DataFrame(),
            "total_missing_cells": 0,
            "duplicate_summary": {
                "duplicate_count": 0,
                "duplicate_percentage": 0.0,
                "has_duplicates": False,
                "duplicate_indices": []
            },
            "infinite_counts": {},
            "constant_columns": [],
            "column_types": {}
        }

    total_rows = len(df)
    analysis_cols = [c for c in df.columns if c not in (RECORD_ID_COL, ORIGINAL_ROW_COL)]
    total_cols = len(analysis_cols)

    # Column typing
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    column_types: Dict[str, str] = {}

    for c in analysis_cols:
        dtype = df[c].dtype
        column_types[str(c)] = str(dtype)
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(str(c))
        else:
            categorical_cols.append(str(c))

    # Missing values
    missing_summary = get_missing_value_summary(df)
    total_missing_cells = int(missing_summary["Missing Count"].sum()) if not missing_summary.empty else 0

    # Duplicates
    dup_summary = get_duplicate_summary(df)

    # Infinite values & constant columns
    infinite_counts: Dict[str, int] = {}
    constant_columns: List[str] = []

    for c in numeric_cols:
        s_numeric = pd.to_numeric(df[c], errors="coerce")
        inf_count = int(np.isinf(s_numeric).sum())
        if inf_count > 0:
            infinite_counts[str(c)] = inf_count

        # Constant check (excluding NaNs and Infs)
        finite_vals = s_numeric[np.isfinite(s_numeric)]
        if len(finite_vals.unique()) <= 1 and len(finite_vals) > 0:
            constant_columns.append(str(c))

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "total_missing_cells": total_missing_cells,
        "duplicate_summary": dup_summary,
        "infinite_counts": infinite_counts,
        "constant_columns": constant_columns,
        "column_types": column_types
    }
