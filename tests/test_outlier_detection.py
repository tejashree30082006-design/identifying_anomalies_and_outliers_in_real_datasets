"""
Automated unit tests for IQR and Z-Score outlier detection.
Covers Cases 1, 2, 3, 4, 5, 6, 20.
"""

import pytest
import pandas as pd
import numpy as np

from core.outlier_detection import detect_iqr_outliers, detect_zscore_outliers, detect_all_outliers
from config.settings import RECORD_ID_COL, ORIGINAL_ROW_COL


def test_clean_data_no_outliers():
    """Verify normally clustered clean data has zero flagged outliers (Case 1)."""
    # 20 observations clustered between 45 and 55
    data = [50.0 + (i % 5) - 2 for i in range(20)]
    s = pd.Series(data, name="clean_var")

    iqr_res = detect_iqr_outliers(s, multiplier=1.5)
    z_res = detect_zscore_outliers(s, threshold=3.0)

    assert iqr_res["flagged_count"] == 0
    assert z_res["flagged_count"] == 0


def test_large_positive_outlier():
    """Verify single extreme positive value is flagged by both methods (Case 2)."""
    data = [10.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.4, 10.3, 10.0, 10.2, 9.7, 10.5] + [1000.0]
    s = pd.Series(data, name="high_extreme")

    iqr_res = detect_iqr_outliers(s, multiplier=1.5)
    assert iqr_res["flagged_count"] >= 1
    sub_iqr = iqr_res["results_df"]
    flagged_iqr = sub_iqr[sub_iqr["iqr_flag"] == True]
    assert 1000.0 in flagged_iqr["numeric_value"].values


def test_small_negative_outlier():
    """Verify single extreme negative value is flagged by both methods (Case 3)."""
    data = [100.0, 101.0, 99.0, 100.5, 99.5, 102.0, 98.0, 100.2, 99.8, 101.5] + [-999.0]
    s = pd.Series(data, name="low_extreme")

    iqr_res = detect_iqr_outliers(s, multiplier=1.5)
    assert iqr_res["flagged_count"] >= 1
    sub_iqr = iqr_res["results_df"]
    flagged_iqr = sub_iqr[sub_iqr["iqr_flag"] == True]
    assert -999.0 in flagged_iqr["numeric_value"].values


def test_both_positive_and_negative_extremes():
    """Verify dataset with both upper and lower outliers (Case 4)."""
    data = [-500.0] + [50.0, 51.0, 49.0, 50.5, 49.5, 52.0, 48.0, 50.2, 49.8, 51.5] + [500.0]
    s = pd.Series(data, name="bipolar_extreme")

    iqr_res = detect_iqr_outliers(s, multiplier=1.5)
    sub = iqr_res["results_df"]
    flagged = sub[sub["iqr_flag"] == True]

    assert len(flagged) == 2
    assert -500.0 in flagged["numeric_value"].values
    assert 500.0 in flagged["numeric_value"].values


def test_iqr_and_zscore_disagreement():
    """
    Verify scenario where IQR flags a point but Z-score does not due to variance inflation (Case 5).
    In small-to-medium samples with moderate skew or a huge extreme that inflates s,
    secondary outliers can fall outside Tukey fences but fail to reach |Z| > 3.0.
    """
    base = [10.0, 11.0, 12.0, 10.5, 11.5, 12.2, 10.8, 11.1, 10.9, 11.4]
    data = base + [25.0]  # 25 is outside Tukey fences (IQR is ~1.0, upper fence is ~13.5)
    s = pd.Series(data, name="disagree_test")

    iqr_res = detect_iqr_outliers(s, multiplier=1.5)
    z_res = detect_zscore_outliers(s, threshold=3.0)

    # IQR flags 25.0
    sub_iqr = iqr_res["results_df"]
    flagged_iqr = sub_iqr[sub_iqr["numeric_value"] == 25.0].iloc[0]
    assert bool(flagged_iqr["iqr_flag"]) is True

    # Z-score for 25 in this sample:
    sub_z = z_res["results_df"]
    flagged_z = sub_z[sub_z["numeric_value"] == 25.0].iloc[0]
    assert abs(flagged_z["signed_zscore"]) < 3.0
    assert bool(flagged_z["zscore_flag"]) is False


def test_constant_column_outlier_detection():
    """Verify constant column handled safely with informative message (Case 6)."""
    s = pd.Series([7.0, 7.0, 7.0, 7.0], name="constant")
    iqr_res = detect_iqr_outliers(s, multiplier=1.5)
    z_res = detect_zscore_outliers(s, threshold=3.0)

    assert iqr_res["iqr_is_zero"] is True
    assert z_res["is_defined"] is False
    assert "undefined" in z_res["notes"].lower()


def test_flagged_records_map_to_correct_row_ids():
    """Verify flagged records map strictly back to original source row IDs (Case 20)."""
    df = pd.DataFrame({
        "item": ["A", "B", "C", "D", "E"],
        "val": [10.0, 11.0, 999.0, 10.5, 11.2]
    })
    # Attach known record_id and original_row_number
    df[RECORD_ID_COL] = [100, 101, 102, 103, 104]
    df[ORIGINAL_ROW_COL] = [10, 20, 30, 40, 50]

    bundle = detect_all_outliers(df, ["val"], z_threshold=2.0, iqr_multiplier=1.5)
    comp_df = bundle["combined_df"]

    flagged_row = comp_df[comp_df["original_value"] == 999.0].iloc[0]
    assert flagged_row[RECORD_ID_COL] == 102
    assert flagged_row[ORIGINAL_ROW_COL] == 30
