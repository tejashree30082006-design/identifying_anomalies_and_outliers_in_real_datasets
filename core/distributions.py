"""
Probability distribution analysis module.
Assesses empirical distribution characteristics (symmetry, skewness, tail weight, kurtosis)
and generates descriptive distribution summaries without assuming normality.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

from utils.validation import sanitize_numeric_series


def analyze_distribution(series: pd.Series) -> Dict[str, Any]:
    """
    Evaluate empirical distribution metrics and shape characteristics.
    Does not assume normal distribution. Identifies skewness and tail behavior.
    """
    valid_s = sanitize_numeric_series(series)
    n = len(valid_s)

    if n == 0:
        return {
            "valid_count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "symmetry_status": "Undefined (No valid observations)",
            "tail_status": "Undefined",
            "is_constant": False,
            "fit_normal_meaningful": False,
            "shape_summary": "No finite numeric data available for distribution analysis."
        }

    vals = valid_s.to_numpy(dtype=float)
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
    median_val = float(np.median(vals))
    is_constant = bool(np.min(vals) == np.max(vals))

    if is_constant:
        return {
            "valid_count": n,
            "mean": mean_val,
            "std": 0.0,
            "median": median_val,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "symmetry_status": "Degenerate / Constant",
            "tail_status": "No variation",
            "is_constant": True,
            "fit_normal_meaningful": False,
            "shape_summary": "Variable has zero variance (constant value). Distributional modeling is inapplicable."
        }

    # Skewness assessment
    skew_val = float(stats.skew(vals, bias=False)) if n >= 3 and std_val > 0 else np.nan
    if np.isnan(skew_val):
        symmetry_status = "Insufficient sample size (N < 3)"
    elif abs(skew_val) < 0.5:
        symmetry_status = "Approximately Symmetric"
    elif 0.5 <= skew_val < 1.0:
        symmetry_status = "Moderately Right-Skewed (Positive Skew)"
    elif skew_val >= 1.0:
        symmetry_status = "Highly Right-Skewed (Positive Skew)"
    elif -1.0 < skew_val <= -0.5:
        symmetry_status = "Moderately Left-Skewed (Negative Skew)"
    else:
        symmetry_status = "Highly Left-Skewed (Negative Skew)"

    # Kurtosis assessment (Fisher's excess kurtosis, normal = 0.0)
    kurt_val = float(stats.kurtosis(vals, fisher=True, bias=False)) if n >= 4 and std_val > 0 else np.nan
    if np.isnan(kurt_val):
        tail_status = "Insufficient sample size (N < 4)"
    elif -0.5 <= kurt_val <= 0.5:
        tail_status = "Mesokurtic (Tails comparable to normal distribution)"
    elif kurt_val > 0.5:
        tail_status = "Leptokurtic (Heavy tails, higher propensity for extreme values)"
    else:
        tail_status = "Platykurtic (Light tails, flatter distribution than normal)"

    fit_normal_meaningful = bool(n >= 8 and not is_constant and std_val > 0)

    # Plain-English summary
    shape_parts = [f"Sample consists of N={n} valid observations."]
    if not np.isnan(skew_val):
        shape_parts.append(f"Empirical skewness is {skew_val:.3f} ({symmetry_status.lower()}).")
    if not np.isnan(kurt_val):
        shape_parts.append(f"Fisher's excess kurtosis is {kurt_val:.3f} ({tail_status.lower()}).")
    if abs(mean_val - median_val) > (0.2 * std_val) if std_val > 0 else False:
        shape_parts.append(f"Mean ({mean_val:.2f}) diverges from median ({median_val:.2f}), reflecting distributional asymmetry.")
    else:
        shape_parts.append(f"Mean ({mean_val:.2f}) and median ({median_val:.2f}) are relatively close.")

    shape_summary = " ".join(shape_parts)

    return {
        "valid_count": n,
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "symmetry_status": symmetry_status,
        "tail_status": tail_status,
        "is_constant": False,
        "fit_normal_meaningful": fit_normal_meaningful,
        "shape_summary": shape_summary
    }


def create_distribution_summary(series: pd.Series) -> Dict[str, Any]:
    """
    Generate a high-level summary structure of distribution properties
    suitable for UI cards and analytical report generation.
    """
    analysis = analyze_distribution(series)
    return {
        "valid_n": analysis["valid_count"],
        "mean": analysis["mean"],
        "median": analysis["median"],
        "std": analysis["std"],
        "skewness": analysis["skewness"],
        "kurtosis": analysis["kurtosis"],
        "symmetry": analysis["symmetry_status"],
        "tails": analysis["tail_status"],
        "is_constant": analysis["is_constant"],
        "fit_meaningful": analysis["fit_normal_meaningful"],
        "narrative": analysis["shape_summary"]
    }
