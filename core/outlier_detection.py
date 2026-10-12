"""
Outlier detection module implementing Tukey's IQR rule and standard Z-score method.
Produces transparent, reproducible metrics with stable record traceability
and neutral anomaly classifications.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from config.settings import (
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_Z_THRESHOLD,
    QUANTILE_METHOD,
    RECORD_ID_COL,
    ORIGINAL_ROW_COL,
    CATEGORY_BOTH,
    CATEGORY_IQR_ONLY,
    CATEGORY_ZSCORE_ONLY,
    CATEGORY_NEITHER
)


def detect_iqr_outliers(
    series: pd.Series,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
    record_ids: Optional[pd.Series] = None,
    row_numbers: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Detect outliers using Tukey's Interquartile Range (IQR) rule.
    
    Fences:
      Lower Fence = Q1 - (multiplier * IQR)
      Upper Fence = Q3 + (multiplier * IQR)
      
    Observations strictly below Lower Fence or strictly above Upper Fence are flagged.
    """
    var_name = str(series.name) if series.name is not None else "variable"
    n_total = len(series)

    # Use supplied record IDs or default to 0..N-1
    rec_ids = record_ids if record_ids is not None else pd.Series(range(n_total), index=series.index)
    row_nums = row_numbers if row_numbers is not None else pd.Series([i + 1 for i in range(n_total)], index=series.index)

    # Convert to numeric and filter finite values
    s_numeric = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(s_numeric)
    valid_vals = s_numeric[finite_mask]
    n_valid = len(valid_vals)

    if n_valid == 0:
        return {
            "variable_name": var_name,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "lower_fence": np.nan,
            "upper_fence": np.nan,
            "multiplier": multiplier,
            "total_observations": n_total,
            "valid_count": 0,
            "flagged_count": 0,
            "iqr_is_zero": False,
            "notes": "No valid finite numerical observations available.",
            "results_df": pd.DataFrame()
        }

    val_arr = valid_vals.to_numpy(dtype=float)
    q1 = float(np.quantile(val_arr, 0.25, method=QUANTILE_METHOD))
    q3 = float(np.quantile(val_arr, 0.75, method=QUANTILE_METHOD))
    iqr_val = q3 - q1

    iqr_is_zero = bool(iqr_val == 0.0)
    lower_fence = q1 - (multiplier * iqr_val)
    upper_fence = q3 + (multiplier * iqr_val)

    notes = ""
    if iqr_is_zero:
        notes = (
            "Caution: Interquartile Range (IQR) is zero. Over 50% of values are identical. "
            "Standard Tukey fences collapse to a single point; applying the rule strictly may flag ordinary variation."
        )

    # Evaluate each row in original order
    rows = []
    flagged_count = 0

    for idx in series.index:
        rid = rec_ids.loc[idx]
        rnum = row_nums.loc[idx]
        val = s_numeric.loc[idx]

        is_flagged = False
        fence_status = "Within Fences"
        deviation_mag = 0.0

        if not np.isfinite(val):
            fence_status = "Missing or Non-Finite"
        elif not iqr_is_zero:
            if val < lower_fence:
                is_flagged = True
                fence_status = "Below Lower Fence"
                deviation_mag = lower_fence - val
                flagged_count += 1
            elif val > upper_fence:
                is_flagged = True
                fence_status = "Above Upper Fence"
                deviation_mag = val - upper_fence
                flagged_count += 1
        else:
            # When IQR is 0
            if val != q1:
                fence_status = "Deviates from Constant Median (Zero-IQR)"

        rows.append({
            RECORD_ID_COL: rid,
            ORIGINAL_ROW_COL: rnum,
            "variable": var_name,
            "original_value": series.loc[idx],
            "numeric_value": val if np.isfinite(val) else np.nan,
            "iqr_flag": is_flagged,
            "iqr_status": fence_status,
            "iqr_deviation": deviation_mag
        })

    results_df = pd.DataFrame(rows)

    return {
        "variable_name": var_name,
        "q1": q1,
        "q3": q3,
        "iqr": iqr_val,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "multiplier": multiplier,
        "total_observations": n_total,
        "valid_count": n_valid,
        "flagged_count": flagged_count,
        "iqr_is_zero": iqr_is_zero,
        "notes": notes,
        "results_df": results_df
    }


def detect_zscore_outliers(
    series: pd.Series,
    threshold: float = DEFAULT_Z_THRESHOLD,
    record_ids: Optional[pd.Series] = None,
    row_numbers: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Detect outliers using standard sample Z-score.
    
    Formula:
      Z = (x - mean) / s (using sample standard deviation, ddof=1)
      
    Observations with |Z| > threshold are flagged.
    """
    var_name = str(series.name) if series.name is not None else "variable"
    n_total = len(series)

    rec_ids = record_ids if record_ids is not None else pd.Series(range(n_total), index=series.index)
    row_nums = row_numbers if row_numbers is not None else pd.Series([i + 1 for i in range(n_total)], index=series.index)

    s_numeric = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(s_numeric)
    valid_vals = s_numeric[finite_mask]
    n_valid = len(valid_vals)

    if n_valid < 2:
        return {
            "variable_name": var_name,
            "mean": np.nan,
            "std": np.nan,
            "threshold": threshold,
            "total_observations": n_total,
            "valid_count": n_valid,
            "flagged_count": 0,
            "is_defined": False,
            "notes": "Sample size is insufficient (N < 2) to compute sample standard deviation.",
            "results_df": pd.DataFrame()
        }

    val_arr = valid_vals.to_numpy(dtype=float)
    mean_val = float(np.mean(val_arr))
    std_val = float(np.std(val_arr, ddof=1))

    if std_val == 0.0:
        return {
            "variable_name": var_name,
            "mean": mean_val,
            "std": 0.0,
            "threshold": threshold,
            "total_observations": n_total,
            "valid_count": n_valid,
            "flagged_count": 0,
            "is_defined": False,
            "notes": "Standard deviation is 0.0 (constant variable). Z-scores are undefined.",
            "results_df": pd.DataFrame()
        }

    rows = []
    flagged_count = 0

    for idx in series.index:
        rid = rec_ids.loc[idx]
        rnum = row_nums.loc[idx]
        val = s_numeric.loc[idx]

        is_flagged = False
        z_score = np.nan
        abs_z = np.nan
        direction = "Within Normal Range"
        excess = 0.0

        if np.isfinite(val):
            z_score = float((val - mean_val) / std_val)
            abs_z = abs(z_score)

            if abs_z > threshold:
                is_flagged = True
                flagged_count += 1
                direction = f"Above Mean (+{abs_z:.2f}σ)" if z_score > 0 else f"Below Mean (-{abs_z:.2f}σ)"
                excess = abs_z - threshold
        else:
            direction = "Missing or Non-Finite"

        rows.append({
            RECORD_ID_COL: rid,
            ORIGINAL_ROW_COL: rnum,
            "variable": var_name,
            "original_value": series.loc[idx],
            "numeric_value": val if np.isfinite(val) else np.nan,
            "zscore_flag": is_flagged,
            "signed_zscore": z_score,
            "abs_zscore": abs_z,
            "zscore_direction": direction,
            "zscore_excess": excess
        })

    results_df = pd.DataFrame(rows)

    return {
        "variable_name": var_name,
        "mean": mean_val,
        "std": std_val,
        "threshold": threshold,
        "total_observations": n_total,
        "valid_count": n_valid,
        "flagged_count": flagged_count,
        "is_defined": True,
        "notes": "",
        "results_df": results_df
    }


def detect_all_outliers(
    df: pd.DataFrame,
    selected_columns: List[str],
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER
) -> Dict[str, Any]:
    """
    Run both Tukey IQR and Z-Score outlier detection across all selected columns.
    Generates combined classification into four mutually exclusive categories:
      1. Flagged by Both Methods
      2. Flagged by IQR Only
      3. Flagged by Z-Score Only
      4. Flagged by Neither Method
    
    Produces both observation-level records and variable-level metadata.
    """
    if df is None or not selected_columns:
        return {
            "combined_df": pd.DataFrame(),
            "iqr_meta": {},
            "zscore_meta": {},
            "category_counts": {}
        }

    rec_ids = df[RECORD_ID_COL] if RECORD_ID_COL in df.columns else pd.Series(range(len(df)), index=df.index)
    row_nums = df[ORIGINAL_ROW_COL] if ORIGINAL_ROW_COL in df.columns else pd.Series([i + 1 for i in range(len(df))], index=df.index)

    iqr_meta: Dict[str, Any] = {}
    zscore_meta: Dict[str, Any] = {}
    combined_rows: List[Dict[str, Any]] = []

    for col in selected_columns:
        if col not in df.columns:
            continue

        iqr_res = detect_iqr_outliers(df[col], multiplier=iqr_multiplier, record_ids=rec_ids, row_numbers=row_nums)
        z_res = detect_zscore_outliers(df[col], threshold=z_threshold, record_ids=rec_ids, row_numbers=row_nums)

        iqr_meta[col] = iqr_res
        zscore_meta[col] = z_res

        iqr_sub = iqr_res["results_df"]
        z_sub = z_res["results_df"]

        # Merge results for this variable
        for i in range(len(df)):
            idx = df.index[i]
            r_id = rec_ids.loc[idx]
            r_num = row_nums.loc[idx]
            orig_val = df.loc[idx, col]

            iqr_flag = False
            iqr_status = "Not Analyzed"
            if not iqr_sub.empty:
                iqr_row = iqr_sub.iloc[i]
                iqr_flag = bool(iqr_row["iqr_flag"])
                iqr_status = str(iqr_row["iqr_status"])

            z_flag = False
            signed_z = np.nan
            abs_z = np.nan
            if not z_sub.empty and z_res["is_defined"]:
                z_row = z_sub.iloc[i]
                z_flag = bool(z_row["zscore_flag"])
                signed_z = z_row["signed_zscore"]
                abs_z = z_row["abs_zscore"]

            # Mutual exclusive category assignment
            if iqr_flag and z_flag:
                category = CATEGORY_BOTH
                explanation = (
                    f"Point falls outside IQR fences ({iqr_status}) and exceeds Z-score threshold "
                    f"(|Z|={abs_z:.2f} > {z_threshold}). High statistical divergence across both methods."
                )
            elif iqr_flag and not z_flag:
                category = CATEGORY_IQR_ONLY
                z_text = f"|Z|={abs_z:.2f} <= {z_threshold}" if not np.isnan(abs_z) else "Z-score undefined"
                explanation = (
                    f"Point falls outside IQR fences ({iqr_status}) but does not exceed Z-score threshold "
                    f"({z_text}). Frequently occurs with skewed or heavy-tailed distributions."
                )
            elif not iqr_flag and z_flag:
                category = CATEGORY_ZSCORE_ONLY
                explanation = (
                    f"Point exceeds Z-score threshold (|Z|={abs_z:.2f} > {z_threshold}) but lies within IQR fences. "
                    "May reflect small sample variance or asymmetric tails."
                )
            else:
                category = CATEGORY_NEITHER
                if pd.isna(orig_val):
                    explanation = "Missing observation (evaluated under data quality checks)."
                else:
                    explanation = "Within expected distribution range under selected rules."

            combined_rows.append({
                RECORD_ID_COL: r_id,
                ORIGINAL_ROW_COL: r_num,
                "variable": col,
                "original_value": orig_val,
                "iqr_flag": iqr_flag,
                "zscore_flag": z_flag,
                "signed_zscore": signed_z,
                "abs_zscore": abs_z,
                "iqr_status": iqr_status,
                "combined_category": category,
                "explanation": explanation
            })

    combined_df = pd.DataFrame(combined_rows)

    # Category counts summary
    category_counts = {
        CATEGORY_BOTH: int((combined_df["combined_category"] == CATEGORY_BOTH).sum()) if not combined_df.empty else 0,
        CATEGORY_IQR_ONLY: int((combined_df["combined_category"] == CATEGORY_IQR_ONLY).sum()) if not combined_df.empty else 0,
        CATEGORY_ZSCORE_ONLY: int((combined_df["combined_category"] == CATEGORY_ZSCORE_ONLY).sum()) if not combined_df.empty else 0,
        CATEGORY_NEITHER: int((combined_df["combined_category"] == CATEGORY_NEITHER).sum()) if not combined_df.empty else 0,
    }

    return {
        "combined_df": combined_df,
        "iqr_meta": iqr_meta,
        "zscore_meta": zscore_meta,
        "category_counts": category_counts
    }
