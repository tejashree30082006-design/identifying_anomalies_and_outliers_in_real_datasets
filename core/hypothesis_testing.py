"""
Hypothesis testing module for statistical anomaly detection.
Implements Shapiro-Wilk test for normality and Grubbs' test for a single outlier.
Provides rigorous assumption checks, sample size validations, and neutral academic interpretations.
"""

from typing import Dict, Any, Optional
import math
import pandas as pd
import numpy as np
from scipy import stats

from config.settings import (
    DEFAULT_ALPHA,
    SHAPIRO_MIN_SAMPLE,
    SHAPIRO_MAX_OFFICIAL_SAMPLE,
    GRUBBS_MIN_SAMPLE
)
from utils.validation import sanitize_numeric_series


def test_normality(series: pd.Series, alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    """
    Conduct Shapiro-Wilk test of normality on finite numerical observations.
    
    Hypotheses:
      H0: The sample data is drawn from a normally distributed population.
      H1: The sample data is not drawn from a normally distributed population.
      
    Valid range: N in [3, 5000].
    """
    var_name = str(series.name) if series.name is not None else "variable"
    valid_s = sanitize_numeric_series(series)
    n = len(valid_s)

    base_result: Dict[str, Any] = {
        "test_name": "Shapiro-Wilk Normality Test",
        "variable_name": var_name,
        "sample_size": n,
        "alpha": alpha,
        "statistic": np.nan,
        "p_value": np.nan,
        "is_valid": False,
        "reject_null": False,
        "decision": "Test Inapplicable",
        "interpretation": "",
        "methodological_caveats": ""
    }

    if n < SHAPIRO_MIN_SAMPLE:
        base_result["interpretation"] = (
            f"Sample size (N={n}) is insufficient for Shapiro-Wilk test (minimum N={SHAPIRO_MIN_SAMPLE} required)."
        )
        return base_result

    vals = valid_s.to_numpy(dtype=float)

    # Constant column check
    if np.min(vals) == np.max(vals):
        base_result["interpretation"] = (
            "Variable is constant (zero variance). Normality test is inapplicable."
        )
        return base_result

    # Sample size warning
    sample_size_note = ""
    if n > SHAPIRO_MAX_OFFICIAL_SAMPLE:
        sample_size_note = (
            f"Note: Sample size (N={n}) exceeds 5,000. With very large samples, "
            "Shapiro-Wilk has extremely high statistical power and will reject H0 for trivial, "
            "practically negligible departures from normality."
        )

    try:
        stat, p_val = stats.shapiro(vals)
        stat = float(stat)
        p_val = float(p_val)

        reject_h0 = bool(p_val < alpha)
        decision = "Reject Null Hypothesis (Evidence against normality)" if reject_h0 else "Fail to Reject Null Hypothesis (Consistent with normality)"

        if reject_h0:
            interp = (
                f"At significance level α={alpha:.2f}, the p-value ({p_val:.4e}) is less than α. "
                "There is statistically significant evidence to reject the null hypothesis of normality. "
                "Distributional non-normality may diminish the reliability of standard Z-score thresholds."
            )
        else:
            interp = (
                f"At significance level α={alpha:.2f}, the p-value ({p_val:.4f}) is greater than or equal to α. "
                "There is insufficient statistical evidence to reject normality. "
                "CRITICAL: Failing to reject H0 does NOT prove that the underlying population is strictly normal."
            )

        base_result.update({
            "statistic": stat,
            "p_value": p_val,
            "is_valid": True,
            "reject_null": reject_h0,
            "decision": decision,
            "interpretation": interp,
            "methodological_caveats": sample_size_note
        })
        return base_result

    except Exception as e:
        base_result["interpretation"] = f"Calculation error during Shapiro-Wilk test: {str(e)}"
        return base_result


def test_grubbs_outlier(series: pd.Series, alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    """
    Grubbs' test for a single suspected outlier in an approximately normally distributed sample.
    
    Hypotheses:
      H0: There are no outliers in the sample.
      H1: Exactly one observation is an outlier (at either minimum or maximum).
      
    Assumptions:
      1. The underlying population is normally distributed.
      2. The test is strictly designed to evaluate at most ONE suspect observation.
    """
    var_name = str(series.name) if series.name is not None else "variable"
    valid_s = sanitize_numeric_series(series)
    n = len(valid_s)

    base_result: Dict[str, Any] = {
        "test_name": "Grubbs' Single-Outlier Test (Two-Sided)",
        "variable_name": var_name,
        "sample_size": n,
        "alpha": alpha,
        "statistic_g": np.nan,
        "critical_value": np.nan,
        "p_value_approx": np.nan,
        "suspect_value": np.nan,
        "suspect_direction": "",
        "is_valid": False,
        "reject_null": False,
        "decision": "Test Inapplicable",
        "interpretation": "",
        "assumptions": (
            "Requires underlying normality and tests for at most ONE outlier. "
            "Masking occurs if multiple extreme values are present."
        )
    }

    if n < GRUBBS_MIN_SAMPLE:
        base_result["interpretation"] = (
            f"Sample size (N={n}) is insufficient for Grubbs' test (minimum N={GRUBBS_MIN_SAMPLE} required)."
        )
        return base_result

    vals = valid_s.to_numpy(dtype=float)
    s_std = float(np.std(vals, ddof=1))
    s_mean = float(np.mean(vals))

    if s_std == 0.0:
        base_result["interpretation"] = (
            "Sample standard deviation is zero (constant variable). Outlier test is inapplicable."
        )
        return base_result

    # Compute deviations
    devs = np.abs(vals - s_mean)
    max_idx = int(np.argmax(devs))
    max_dev = float(devs[max_idx])
    suspect_val = float(vals[max_idx])
    suspect_dir = "Upper Extreme (Maximum)" if suspect_val > s_mean else "Lower Extreme (Minimum)"

    # Grubbs' G statistic
    g_stat = float(max_dev / s_std)

    # Critical value calculation using Student's t distribution
    # df = N - 2
    # p_t = 1 - alpha / (2 * N)
    df = n - 2
    p_t = 1.0 - (alpha / (2.0 * n))
    t_crit = float(stats.t.ppf(p_t, df))

    numerator = (n - 1) * math.sqrt(t_crit ** 2)
    denominator = math.sqrt(n) * math.sqrt(df + (t_crit ** 2))
    g_crit = float(numerator / denominator)

    # Approximate p-value using Bonferroni bounds
    # t_stat corresponding to G
    # G^2 * (N - 2) / ( (N - 1)^2 - N * G^2 )
    term = ((n - 1) ** 2) - (n * (g_stat ** 2))
    if term > 0:
        t_stat = math.sqrt((g_stat ** 2) * df / term)
        p_single_tail = float(1.0 - stats.t.cdf(t_stat, df))
        p_val_approx = min(1.0, max(0.0, 2.0 * n * p_single_tail))
    else:
        p_val_approx = 0.0

    reject_h0 = bool(g_stat > g_crit)
    decision = "Reject Null Hypothesis (Suspect point is a statistical outlier)" if reject_h0 else "Fail to Reject Null Hypothesis (No statistically significant single outlier)"

    if reject_h0:
        interp = (
            f"At α={alpha:.2f}, Grubbs' test statistic G = {g_stat:.4f} exceeds critical threshold G_crit = {g_crit:.4f} "
            f"(p ≈ {p_val_approx:.4e}). The extreme observation ({suspect_val:.2f}, {suspect_dir}) is a statistically "
            "significant outlier under the assumption of sample normality. "
            "Note: This does not prove intentional fabrication or data fraud."
        )
    else:
        interp = (
            f"At α={alpha:.2f}, Grubbs' test statistic G = {g_stat:.4f} does not exceed critical threshold G_crit = {g_crit:.4f} "
            f"(p ≈ {p_val_approx:.4f}). There is insufficient statistical evidence that the extreme observation ({suspect_val:.2f}) "
            "is an outlier from the assumed normal distribution."
        )

    base_result.update({
        "statistic_g": g_stat,
        "critical_value": g_crit,
        "p_value_approx": p_val_approx,
        "suspect_value": suspect_val,
        "suspect_direction": suspect_dir,
        "is_valid": True,
        "reject_null": reject_h0,
        "decision": decision,
        "interpretation": interp
    })

    return base_result
