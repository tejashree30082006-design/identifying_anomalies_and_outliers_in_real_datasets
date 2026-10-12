"""
Automated unit tests for descriptive statistics module.
Covers Cases 1, 6, 7, 8, 9, 10, 14.
"""

import pytest
import pandas as pd
import numpy as np

from core.descriptive_stats import calculate_descriptive_statistics


def test_clean_descriptive_statistics():
    """Verify standard metrics match known formulas on clean data (Case 1)."""
    # Sample: [10, 20, 30, 40, 50]
    # Mean = 30.0, Median = 30.0, Min = 10, Max = 50, Range = 40
    # Sample Variance (ddof=1) = ((400 + 100 + 0 + 100 + 400)/4) = 250.0
    # Sample Std = sqrt(250) ≈ 15.8114
    # Q1 = 20.0, Q3 = 40.0, IQR = 20.0
    # Skewness = 0.0 (symmetric)
    df = pd.DataFrame({"vals": [10.0, 20.0, 30.0, 40.0, 50.0]})
    res = calculate_descriptive_statistics(df, ["vals"])

    assert res.loc["Valid Observations", "vals"] == 5
    assert res.loc["Missing Count", "vals"] == 0
    assert pytest.approx(res.loc["Mean", "vals"], 0.001) == 30.0
    assert pytest.approx(res.loc["Median", "vals"], 0.001) == 30.0
    assert pytest.approx(res.loc["Minimum", "vals"], 0.001) == 10.0
    assert pytest.approx(res.loc["Maximum", "vals"], 0.001) == 50.0
    assert pytest.approx(res.loc["Range", "vals"], 0.001) == 40.0
    assert pytest.approx(res.loc["Sample Variance (s², ddof=1)", "vals"], 0.001) == 250.0
    assert pytest.approx(res.loc["Sample Std Dev (s, ddof=1)", "vals"], 0.001) == np.sqrt(250.0)
    assert pytest.approx(res.loc["Q1 (25th Percentile)", "vals"], 0.001) == 20.0
    assert pytest.approx(res.loc["Q3 (75th Percentile)", "vals"], 0.001) == 40.0
    assert pytest.approx(res.loc["IQR (Q3 - Q1)", "vals"], 0.001) == 20.0
    assert pytest.approx(res.loc["Skewness (Sample)", "vals"], 0.001) == 0.0


def test_constant_column():
    """Verify safe handling of zero-variance constant column (Case 6)."""
    df = pd.DataFrame({"const": [42.0, 42.0, 42.0, 42.0]})
    res = calculate_descriptive_statistics(df, ["const"])

    assert res.loc["Mean", "const"] == 42.0
    assert res.loc["Sample Variance (s², ddof=1)", "const"] == 0.0
    assert res.loc["Sample Std Dev (s, ddof=1)", "const"] == 0.0
    assert res.loc["IQR (Q3 - Q1)", "const"] == 0.0
    assert "Constant" in res.loc["Status / Notes", "const"]


def test_missing_and_infinite_values():
    """Verify non-finite and missing values are excluded from estimates (Cases 7, 8)."""
    df = pd.DataFrame({"mixed": [10.0, np.nan, 20.0, np.inf, 30.0, -np.inf]})
    res = calculate_descriptive_statistics(df, ["mixed"])

    assert res.loc["Total Observations", "mixed"] == 6
    assert res.loc["Valid Observations", "mixed"] == 3
    assert res.loc["Missing Count", "mixed"] == 1
    assert res.loc["Infinite Count", "mixed"] == 2
    assert pytest.approx(res.loc["Mean", "mixed"], 0.001) == 20.0


def test_all_missing_column():
    """Verify all-missing column returns NaN metrics without throwing errors (Case 9)."""
    df = pd.DataFrame({"all_na": [np.nan, np.nan, np.nan]})
    res = calculate_descriptive_statistics(df, ["all_na"])

    assert res.loc["Valid Observations", "all_na"] == 0
    assert pd.isna(res.loc["Mean", "all_na"])
    assert "All observations missing" in res.loc["Status / Notes", "all_na"]


def test_small_sample_size():
    """Verify small sample sizes (N=1, N=2) do not raise errors (Case 14)."""
    df_single = pd.DataFrame({"single": [100.0]})
    res1 = calculate_descriptive_statistics(df_single, ["single"])
    assert res1.loc["Valid Observations", "single"] == 1
    assert pd.isna(res1.loc["Sample Variance (s², ddof=1)", "single"])

    df_two = pd.DataFrame({"pair": [10.0, 20.0]})
    res2 = calculate_descriptive_statistics(df_two, ["pair"])
    assert res2.loc["Valid Observations", "pair"] == 2
    assert pytest.approx(res2.loc["Sample Variance (s², ddof=1)", "pair"], 0.01) == 50.0
    # Skewness undefined for N < 3
    assert pd.isna(res2.loc["Skewness (Sample)", "pair"])
