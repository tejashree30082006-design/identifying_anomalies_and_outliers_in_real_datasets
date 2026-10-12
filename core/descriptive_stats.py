"""
Descriptive statistics module.
Calculates summary metrics, dispersion measures, quartiles, skewness, and kurtosis
with strict handling of missing values, infinities, small samples, and constant columns.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

from config.settings import QUANTILE_METHOD


def calculate_descriptive_statistics(df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """
    Calculate comprehensive descriptive statistics for selected numerical columns.
    Excludes non-finite and missing values for point estimates while reporting counts.
    
    Returns a DataFrame where rows represent statistical metrics and columns are variables.
    """
    if df is None or not selected_columns:
        return pd.DataFrame()

    results: Dict[str, Dict[str, Any]] = {}

    for col in selected_columns:
        if col not in df.columns:
            continue

        raw_series = df[col]
        total_obs = len(raw_series)
        missing_count = int(raw_series.isna().sum())

        # Extract finite numeric values
        s_numeric = pd.to_numeric(raw_series, errors="coerce")
        finite_mask = np.isfinite(s_numeric)
        valid_series = s_numeric[finite_mask].dropna()
        n_valid = len(valid_series)
        inf_count = int(np.isinf(s_numeric).sum())

        col_stats: Dict[str, Any] = {
            "Total Observations": total_obs,
            "Valid Observations": n_valid,
            "Missing Count": missing_count,
            "Infinite Count": inf_count,
            "Mean": np.nan,
            "Median": np.nan,
            "Mode": "N/A",
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Range": np.nan,
            "Sample Variance (s², ddof=1)": np.nan,
            "Sample Std Dev (s, ddof=1)": np.nan,
            "Q1 (25th Percentile)": np.nan,
            "Q3 (75th Percentile)": np.nan,
            "IQR (Q3 - Q1)": np.nan,
            "Skewness (Sample)": np.nan,
            "Kurtosis (Fisher's Excess)": np.nan,
            "Status / Notes": "Normal"
        }

        if n_valid == 0:
            col_stats["Status / Notes"] = "All observations missing or non-finite"
            results[col] = col_stats
            continue

        # Point estimates
        val_arr = valid_series.to_numpy(dtype=float)
        mean_val = float(np.mean(val_arr))
        median_val = float(np.median(val_arr))
        min_val = float(np.min(val_arr))
        max_val = float(np.max(val_arr))
        range_val = max_val - min_val

        col_stats["Mean"] = mean_val
        col_stats["Median"] = median_val
        col_stats["Minimum"] = min_val
        col_stats["Maximum"] = max_val
        col_stats["Range"] = range_val

        # Mode calculation
        try:
            val_counts = valid_series.value_counts()
            if len(val_counts) == n_valid and n_valid > 1:
                col_stats["Mode"] = "All values unique"
            elif not val_counts.empty:
                top_count = val_counts.iloc[0]
                modes = val_counts[val_counts == top_count].index.tolist()
                if len(modes) == 1:
                    col_stats["Mode"] = float(modes[0])
                elif len(modes) <= 3:
                    col_stats["Mode"] = ", ".join(f"{m:.2f}" for m in modes)
                else:
                    col_stats["Mode"] = "Multimodal (>3 modes)"
        except Exception:
            col_stats["Mode"] = "N/A"

        # Quartiles
        q1 = float(np.quantile(val_arr, 0.25, method=QUANTILE_METHOD))
        q3 = float(np.quantile(val_arr, 0.75, method=QUANTILE_METHOD))
        iqr_val = q3 - q1

        col_stats["Q1 (25th Percentile)"] = q1
        col_stats["Q3 (75th Percentile)"] = q3
        col_stats["IQR (Q3 - Q1)"] = iqr_val

        # Dispersion (requires N >= 2 for sample variance)
        if n_valid >= 2:
            s_var = float(np.var(val_arr, ddof=1))
            s_std = float(np.std(val_arr, ddof=1))
            col_stats["Sample Variance (s², ddof=1)"] = s_var
            col_stats["Sample Std Dev (s, ddof=1)"] = s_std
        else:
            col_stats["Status / Notes"] = "Insufficient sample size for variance (N < 2)"

        # Skewness (Sample unbiased requires N >= 3)
        if n_valid >= 3:
            if s_std == 0:
                col_stats["Skewness (Sample)"] = 0.0
            else:
                try:
                    col_stats["Skewness (Sample)"] = float(stats.skew(val_arr, bias=False))
                except Exception:
                    col_stats["Skewness (Sample)"] = np.nan
        else:
            col_stats["Skewness (Sample)"] = np.nan

        # Kurtosis (Fisher's excess, sample unbiased requires N >= 4)
        if n_valid >= 4:
            if s_std == 0:
                col_stats["Kurtosis (Fisher's Excess)"] = 0.0
            else:
                try:
                    col_stats["Kurtosis (Fisher's Excess)"] = float(stats.kurtosis(val_arr, fisher=True, bias=False))
                except Exception:
                    col_stats["Kurtosis (Fisher's Excess)"] = np.nan
        else:
            col_stats["Kurtosis (Fisher's Excess)"] = np.nan

        # Constant check
        if min_val == max_val:
            col_stats["Status / Notes"] = "Constant column (Zero variance)"

        results[col] = col_stats

    # Return DataFrame with metrics as index and columns as variables
    summary_df = pd.DataFrame(results)
    return summary_df
