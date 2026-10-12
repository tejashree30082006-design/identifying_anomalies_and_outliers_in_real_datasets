"""
Automated unit tests for export utilities and report generation.
Covers Case 18.
"""

import json
import pytest
import pandas as pd
import numpy as np

from utils.export_utils import (
    sanitize_for_json,
    export_dataframe_to_csv,
    export_flagged_records_csv,
    export_analysis_to_json
)
from core.report_generator import generate_markdown_report


def test_sanitize_for_json():
    """Verify conversion of NaN, Inf, and NumPy types to JSON-safe representations (Case 18)."""
    raw_dict = {
        "nan_val": np.nan,
        "pos_inf": np.inf,
        "neg_inf": -np.inf,
        "int_num": np.int64(42),
        "float_num": np.float64(3.1415),
        "nested": {
            "more_nan": float("nan")
        }
    }
    safe_dict = sanitize_for_json(raw_dict)
    # Must serialize via json.dumps without error
    json_text = json.dumps(safe_dict)
    parsed = json.loads(json_text)

    assert parsed["nan_val"] is None
    assert parsed["pos_inf"] == "Infinity"
    assert parsed["neg_inf"] == "-Infinity"
    assert parsed["int_num"] == 42
    assert pytest.approx(parsed["float_num"], 0.001) == 3.1415
    assert parsed["nested"]["more_nan"] is None


def test_csv_exports():
    """Verify CSV string generation for dataframes and flagged records (Case 18)."""
    comp_df = pd.DataFrame({
        "_original_row_number": [1, 2, 3],
        "variable": ["val", "val", "val"],
        "original_value": [10.0, 999.0, 15.0],
        "iqr_flag": [False, True, False],
        "zscore_flag": [False, True, False]
    })

    # Flagged records export
    flagged_csv = export_flagged_records_csv(comp_df)
    assert "999.0" in flagged_csv
    assert "10.0" not in flagged_csv  # Row 1 is not flagged

    # Full table export
    full_csv = export_dataframe_to_csv(comp_df)
    assert "10.0" in full_csv
    assert "999.0" in full_csv


def test_markdown_report_generation():
    """Verify generated Markdown report contains essential sections, settings, and disclaimers (Case 18)."""
    mock_bundle = {
        "dataset_name": "Test_Dataset.csv",
        "selected_columns": ["metric_a"],
        "iqr_multiplier": 1.5,
        "z_threshold": 3.0,
        "alpha": 0.05,
        "quality": {
            "total_rows": 100,
            "total_columns": 5,
            "numeric_columns": ["metric_a"],
            "categorical_columns": ["cat_b"],
            "total_missing_cells": 2,
            "duplicate_summary": {"duplicate_count": 1, "duplicate_percentage": 1.0},
            "infinite_counts": {},
            "constant_columns": []
        },
        "desc_stats": pd.DataFrame({"metric_a": {"Mean": 50.0, "Sample Std Dev (s, ddof=1)": 10.0}}),
        "outlier_summary": {
            "category_counts": {
                "Flagged by Both Methods": 2,
                "Flagged by IQR Only": 1,
                "Flagged by Z-Score Only": 0,
                "Flagged by Neither Method": 97
            },
            "unique_records_summary": {
                "total_unique_records": 100,
                "flagged_unique_records": 3,
                "flagged_record_percentage": 3.0,
                "unflagged_unique_records": 97
            }
        },
        "normality_tests": {
            "metric_a": {
                "is_valid": True,
                "statistic": 0.985,
                "p_value": 0.234,
                "decision": "Fail to Reject Null",
                "interpretation": "Consistent with normality"
            }
        },
        "grubbs_tests": {}
    }

    report = generate_markdown_report(mock_bundle)
    assert "Statistical Data Anomaly & Outlier Audit Report" in report
    assert "Methodological Notice & Scientific Disclaimer" in report
    assert "Test_Dataset.csv" in report
    assert "Total Records:" in report
    assert "Flagged by Both Methods" in report
