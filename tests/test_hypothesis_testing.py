"""
Automated unit tests for hypothesis testing module.
Covers Cases 15, 16.
"""

import pytest
import pandas as pd
import numpy as np

# Import under non-test_ prefixes so pytest test collector does not treat library functions as test cases
from core.hypothesis_testing import (
    test_normality as calc_normality,
    test_grubbs_outlier as calc_grubbs
)


def test_shapiro_sample_sizes():
    """Verify Shapiro-Wilk enforces sample size bounds (N >= 3) (Case 15)."""
    # N=2 is invalid
    s_small = pd.Series([1.0, 2.0])
    res_small = calc_normality(s_small)
    assert res_small["is_valid"] is False
    assert "insufficient" in res_small["interpretation"].lower()

    # N=3 is valid
    s_valid = pd.Series([10.0, 15.0, 20.0])
    res_valid = calc_normality(s_valid)
    assert res_valid["is_valid"] is True
    assert 0.0 <= res_valid["p_value"] <= 1.0


def test_normally_distributed_sample():
    """Verify normal sample fails to reject null hypothesis at alpha=0.05 (Case 16)."""
    np.random.seed(42)
    # 50 samples from true Gaussian distribution
    gaussian_sample = pd.Series(np.random.normal(loc=100.0, scale=15.0, size=50), name="normal_var")
    res = calc_normality(gaussian_sample, alpha=0.05)

    assert res["is_valid"] is True
    assert res["p_value"] > 0.05
    assert res["reject_null"] is False
    assert "Fail to Reject" in res["decision"]


def test_strongly_non_normal_sample():
    """Verify heavily skewed or non-Gaussian sample rejects null hypothesis (Case 16)."""
    np.random.seed(42)
    # 60 samples from heavy Pareto / exponential distribution
    skewed_sample = pd.Series(np.random.exponential(scale=10.0, size=60) ** 3, name="skewed_var")
    res = calc_normality(skewed_sample, alpha=0.05)

    assert res["is_valid"] is True
    assert res["p_value"] < 0.05
    assert res["reject_null"] is True
    assert "Reject Null" in res["decision"]


def test_grubbs_outlier_test():
    """Verify Grubbs' test identifies single statistically significant outlier (Case 15)."""
    np.random.seed(123)
    data = list(np.random.normal(50.0, 2.0, 25)) + [150.0]  # 150.0 is ~50 std dev above mean
    s = pd.Series(data, name="grubbs_var")

    g_res = calc_grubbs(s, alpha=0.05)
    assert g_res["is_valid"] is True
    assert g_res["statistic_g"] > g_res["critical_value"]
    assert g_res["reject_null"] is True
    assert pytest.approx(g_res["suspect_value"], 0.1) == 150.0
    assert "Upper Extreme" in g_res["suspect_direction"]


def test_grubbs_constant_and_small_sample():
    """Verify Grubbs handles constant data and small sample sizes safely (Case 6, 15)."""
    # Small N
    res_small = calc_grubbs(pd.Series([1.0, 2.0]))
    assert res_small["is_valid"] is False

    # Constant
    res_const = calc_grubbs(pd.Series([5.0, 5.0, 5.0, 5.0]))
    assert res_const["is_valid"] is False
    assert "zero" in res_const["interpretation"].lower()
