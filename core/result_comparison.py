"""
Result comparison module.
Integrates Tukey IQR and Z-Score detection results into four mutually exclusive categories.
Prevents double-counting and provides both observation-level and unique-record summaries.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from config.settings import (
    RECORD_ID_COL,
    ORIGINAL_ROW_COL,
    CATEGORY_BOTH,
    CATEGORY_IQR_ONLY,
    CATEGORY_ZSCORE_ONLY,
    CATEGORY_NEITHER
)


def compare_detection_methods(
    iqr_results: Union[Dict[str, Any], pd.DataFrame],
    zscore_results: Union[Dict[str, Any], pd.DataFrame]
) -> Dict[str, Any]:
    """
    Compare and synthesize IQR and Z-score outlier detection outcomes.
    
    Accepts:
      - Two single-variable result dictionaries (from detect_iqr_outliers & detect_zscore_outliers), OR
      - Two DataFrames already containing detection outputs.
      
    Returns a unified comparison dictionary with observation-level records,
    four mutually exclusive category tallies, and unique-record summaries.
    """
    # Extract DataFrames
    iqr_df: pd.DataFrame
    z_df: pd.DataFrame

    if isinstance(iqr_results, dict):
        iqr_df = iqr_results.get("results_df", pd.DataFrame())
    else:
        iqr_df = iqr_results

    if isinstance(zscore_results, dict):
        z_df = zscore_results.get("results_df", pd.DataFrame())
    else:
        z_df = zscore_results

    if iqr_df.empty and z_df.empty:
        return {
            "comparison_df": pd.DataFrame(),
            "category_counts": {
                CATEGORY_BOTH: 0,
                CATEGORY_IQR_ONLY: 0,
                CATEGORY_ZSCORE_ONLY: 0,
                CATEGORY_NEITHER: 0
            },
            "unique_records_summary": {
                "total_unique_records": 0,
                "flagged_unique_records": 0,
                "unflagged_unique_records": 0
            },
            "method_agreement_pct": 100.0,
            "total_pairs": 0
        }

    # Ensure index alignment
    n_rows = max(len(iqr_df), len(z_df))
    rows = []

    for i in range(n_rows):
        has_iqr = i < len(iqr_df)
        has_z = i < len(z_df)

        r_id = iqr_df.iloc[i][RECORD_ID_COL] if has_iqr and RECORD_ID_COL in iqr_df.columns else (
            z_df.iloc[i][RECORD_ID_COL] if has_z and RECORD_ID_COL in z_df.columns else i
        )
        r_num = iqr_df.iloc[i][ORIGINAL_ROW_COL] if has_iqr and ORIGINAL_ROW_COL in iqr_df.columns else (
            z_df.iloc[i][ORIGINAL_ROW_COL] if has_z and ORIGINAL_ROW_COL in z_df.columns else i + 1
        )
        var_name = iqr_df.iloc[i]["variable"] if has_iqr and "variable" in iqr_df.columns else (
            z_df.iloc[i]["variable"] if has_z and "variable" in z_df.columns else "variable"
        )
        orig_val = iqr_df.iloc[i]["original_value"] if has_iqr and "original_value" in iqr_df.columns else (
            z_df.iloc[i]["original_value"] if has_z and "original_value" in z_df.columns else np.nan
        )

        iqr_flag = bool(iqr_df.iloc[i]["iqr_flag"]) if has_iqr and "iqr_flag" in iqr_df.columns else False
        iqr_status = str(iqr_df.iloc[i]["iqr_status"]) if has_iqr and "iqr_status" in iqr_df.columns else "Not Analyzed"

        z_flag = bool(z_df.iloc[i]["zscore_flag"]) if has_z and "zscore_flag" in z_df.columns else False
        signed_z = z_df.iloc[i]["signed_zscore"] if has_z and "signed_zscore" in z_df.columns else np.nan
        abs_z = z_df.iloc[i]["abs_zscore"] if has_z and "abs_zscore" in z_df.columns else np.nan

        # Mutual exclusive categorization
        if iqr_flag and z_flag:
            cat = CATEGORY_BOTH
            expl = f"Divergence confirmed by both methods: outside IQR fences ({iqr_status}) and |Z|={abs_z:.2f}."
        elif iqr_flag and not z_flag:
            cat = CATEGORY_IQR_ONLY
            z_str = f"|Z|={abs_z:.2f}" if not np.isnan(abs_z) else "Z undefined"
            expl = f"Detected by IQR only ({iqr_status}); within Z-score threshold ({z_str})."
        elif not iqr_flag and z_flag:
            cat = CATEGORY_ZSCORE_ONLY
            expl = f"Detected by Z-score only (|Z|={abs_z:.2f}); within Tukey IQR fences."
        else:
            cat = CATEGORY_NEITHER
            if pd.isna(orig_val):
                expl = "Missing or non-finite observation."
            else:
                expl = "Within expected distribution bounds under both detection methods."

        rows.append({
            RECORD_ID_COL: r_id,
            ORIGINAL_ROW_COL: r_num,
            "variable": var_name,
            "original_value": orig_val,
            "iqr_flag": iqr_flag,
            "zscore_flag": z_flag,
            "signed_zscore": signed_z,
            "abs_zscore": abs_z,
            "iqr_status": iqr_status,
            "combined_category": cat,
            "explanation": expl
        })

    comp_df = pd.DataFrame(rows)
    total_pairs = len(comp_df)

    # 4 mutually exclusive categories
    counts = {
        CATEGORY_BOTH: int((comp_df["combined_category"] == CATEGORY_BOTH).sum()),
        CATEGORY_IQR_ONLY: int((comp_df["combined_category"] == CATEGORY_IQR_ONLY).sum()),
        CATEGORY_ZSCORE_ONLY: int((comp_df["combined_category"] == CATEGORY_ZSCORE_ONLY).sum()),
        CATEGORY_NEITHER: int((comp_df["combined_category"] == CATEGORY_NEITHER).sum())
    }

    # Method agreement percentage
    agreed_count = counts[CATEGORY_BOTH] + counts[CATEGORY_NEITHER]
    agreement_pct = round((agreed_count / total_pairs * 100.0), 2) if total_pairs > 0 else 100.0

    # Unique record summary (handles single or multiple variables without double counting)
    unique_rec_ids = comp_df[RECORD_ID_COL].unique()
    total_unique = len(unique_rec_ids)

    # Group by record_id to identify records flagged in any variable
    flagged_mask = (comp_df["combined_category"] != CATEGORY_NEITHER)
    flagged_rec_ids = set(comp_df.loc[flagged_mask, RECORD_ID_COL].unique())
    flagged_unique = len(flagged_rec_ids)
    unflagged_unique = total_unique - flagged_unique

    unique_summary = {
        "total_unique_records": total_unique,
        "flagged_unique_records": flagged_unique,
        "unflagged_unique_records": unflagged_unique,
        "flagged_record_percentage": round((flagged_unique / total_unique * 100.0), 2) if total_unique > 0 else 0.0
    }

    return {
        "comparison_df": comp_df,
        "category_counts": counts,
        "unique_records_summary": unique_summary,
        "method_agreement_pct": agreement_pct,
        "total_pairs": total_pairs
    }
