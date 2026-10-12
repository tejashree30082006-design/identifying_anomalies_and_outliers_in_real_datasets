"""
Automated unit tests for result comparison and classification module.
Covers Case 17.
"""

import pytest
import pandas as pd
import numpy as np

from core.result_comparison import compare_detection_methods
from config.settings import (
    CATEGORY_BOTH,
    CATEGORY_IQR_ONLY,
    CATEGORY_ZSCORE_ONLY,
    CATEGORY_NEITHER,
    RECORD_ID_COL,
    ORIGINAL_ROW_COL
)


def test_mutually_exclusive_categories_no_double_counting():
    """
    Verify the four mutually exclusive categories sum to the exact total number of pairs
    without double counting (Case 17).
    """
    # Create mock IQR and Z-Score results
    # Row 0: Flagged by Both
    # Row 1: Flagged by IQR only
    # Row 2: Flagged by Z-Score only
    # Row 3: Flagged by Neither
    iqr_df = pd.DataFrame({
        RECORD_ID_COL: [0, 1, 2, 3],
        ORIGINAL_ROW_COL: [1, 2, 3, 4],
        "variable": ["val", "val", "val", "val"],
        "original_value": [500.0, 50.0, 60.0, 20.0],
        "iqr_flag": [True, True, False, False],
        "iqr_status": ["Above Fence", "Above Fence", "Within", "Within"]
    })

    z_df = pd.DataFrame({
        RECORD_ID_COL: [0, 1, 2, 3],
        ORIGINAL_ROW_COL: [1, 2, 3, 4],
        "variable": ["val", "val", "val", "val"],
        "original_value": [500.0, 50.0, 60.0, 20.0],
        "zscore_flag": [True, False, True, False],
        "signed_zscore": [4.5, 2.1, 3.2, 0.2],
        "abs_zscore": [4.5, 2.1, 3.2, 0.2]
    })

    comp_res = compare_detection_methods(iqr_df, z_df)
    counts = comp_res["category_counts"]
    total = comp_res["total_pairs"]

    assert total == 4
    assert counts[CATEGORY_BOTH] == 1
    assert counts[CATEGORY_IQR_ONLY] == 1
    assert counts[CATEGORY_ZSCORE_ONLY] == 1
    assert counts[CATEGORY_NEITHER] == 1

    # Exact equality: sum of categories must equal total pairs
    sum_cats = sum(counts.values())
    assert sum_cats == total

    # Unique record summary check
    unique_summary = comp_res["unique_records_summary"]
    assert unique_summary["total_unique_records"] == 4
    # Rows 0, 1, 2 are flagged -> 3 unique records flagged
    assert unique_summary["flagged_unique_records"] == 3
    assert unique_summary["unflagged_unique_records"] == 1
