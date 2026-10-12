"""
Automated unit tests for data quality audit module.
Covers Cases 6, 7, 8, 9, 10, 11, 12.
"""

import pytest
import pandas as pd
import numpy as np

from core.data_quality import analyze_data_quality, get_missing_value_summary, get_duplicate_summary


def test_missing_value_summary():
    """Verify correct calculation of missing counts and percentages (Case 7, 9)."""
    df = pd.DataFrame({
        "complete_col": [1.0, 2.0, 3.0, 4.0],
        "partial_missing": [1.0, np.nan, 3.0, np.nan],
        "all_missing": [np.nan, np.nan, np.nan, np.nan]
    })
    summary = get_missing_value_summary(df)
    assert len(summary) == 3

    partial_row = summary[summary["Column"] == "partial_missing"].iloc[0]
    assert partial_row["Missing Count"] == 2
    assert partial_row["Missing Percentage"] == 50.0
    assert partial_row["Valid Count"] == 2

    all_miss_row = summary[summary["Column"] == "all_missing"].iloc[0]
    assert all_miss_row["Missing Count"] == 4
    assert all_miss_row["Missing Percentage"] == 100.0
    assert all_miss_row["Valid Count"] == 0


def test_duplicate_summary():
    """Verify duplicate detection and percentage calculation (Case 11)."""
    df = pd.DataFrame({
        "name": ["A", "B", "A", "C", "A"],
        "num": [1, 2, 1, 3, 1]
    })
    dup_res = get_duplicate_summary(df)
    assert dup_res["has_duplicates"] is True
    # Rows at index 2 and index 4 are duplicates of index 0
    assert dup_res["duplicate_count"] == 2
    assert dup_res["duplicate_percentage"] == 40.0


def test_infinite_values_and_constant_columns():
    """Verify detection of infinities and constant zero-variance columns (Cases 6, 8)."""
    df = pd.DataFrame({
        "inf_col": [1.0, np.inf, 3.0, -np.inf],
        "const_col": [5.0, 5.0, 5.0, 5.0],
        "normal_col": [10.0, 20.0, 30.0, 40.0]
    })
    audit = analyze_data_quality(df)
    assert audit["infinite_counts"].get("inf_col") == 2
    assert "const_col" in audit["constant_columns"]
    assert "normal_col" not in audit["constant_columns"]


def test_non_numeric_and_empty_dataset():
    """Verify non-numeric column identification and empty dataset handling (Cases 10, 12)."""
    # Non-numeric
    df_mixed = pd.DataFrame({
        "num": [1, 2, 3],
        "text": ["red", "green", "blue"],
        "dt": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    })
    audit_mixed = analyze_data_quality(df_mixed)
    assert "num" in audit_mixed["numeric_columns"]
    assert "text" in audit_mixed["categorical_columns"]

    # Empty
    audit_empty = analyze_data_quality(pd.DataFrame())
    assert audit_empty["total_rows"] == 0
    assert audit_empty["total_columns"] == 0
    assert audit_empty["total_missing_cells"] == 0
