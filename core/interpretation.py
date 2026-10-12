"""
Analytical interpretation module.
Generates dynamic, rule-based, neutral academic interpretations from calculated statistical findings.
Adheres strictly to scientific objectivity: never claims data is fake or fraudulent from outlier status alone.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from config.settings import (
    CATEGORY_BOTH,
    CATEGORY_IQR_ONLY,
    CATEGORY_ZSCORE_ONLY,
    CATEGORY_NEITHER,
    STATISTICAL_DISCLAIMER
)


def generate_interpretation(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate structured, objective interpretations from comprehensive analysis results.
    
    Inputs expected in analysis_results:
      - quality: Output of analyze_data_quality
      - desc_stats: DataFrame from calculate_descriptive_statistics
      - outlier_summary: Output from detect_all_outliers or compare_detection_methods
      - normality_tests: Dict of normality test results per variable (optional)
      - grubbs_tests: Dict of Grubbs test results per variable (optional)
      - selected_columns: List of analyzed variables
    """
    selected_cols = analysis_results.get("selected_columns", [])
    quality = analysis_results.get("quality", {})
    desc_stats = analysis_results.get("desc_stats", pd.DataFrame())
    outlier_summary = analysis_results.get("outlier_summary", {})
    normality_tests = analysis_results.get("normality_tests", {})
    grubbs_tests = analysis_results.get("grubbs_tests", {})

    sections: List[Dict[str, str]] = []

    # 1. Data Integrity and Hygiene
    dq_points = []
    total_rows = quality.get("total_rows", 0)
    total_cols = quality.get("total_columns", 0)
    missing_cells = quality.get("total_missing_cells", 0)
    dup_info = quality.get("duplicate_summary", {})
    dup_count = dup_info.get("duplicate_count", 0)
    inf_counts = quality.get("infinite_counts", {})
    const_cols = quality.get("constant_columns", [])

    dq_points.append(f"The imported dataset contains {total_rows} observations across {total_cols} columns.")
    if missing_cells > 0:
        dq_points.append(
            f"Data-hygiene audit identified {missing_cells} missing values. Missing observations were isolated "
            "and excluded from point calculations under standard complete-case estimation policies."
        )
    else:
        dq_points.append("No missing values were detected; the dataset exhibits complete record coverage.")

    if dup_count > 0:
        dq_points.append(
            f"Identified {dup_count} duplicate records ({dup_info.get('duplicate_percentage', 0.0)}% of dataset). "
            "Stable row identifiers were assigned to preserve row provenance during outlier tracking."
        )

    if inf_counts:
        inf_summary = ", ".join(f"{col} ({cnt})" for col, cnt in inf_counts.items())
        dq_points.append(f"Non-finite / infinite values detected in: {inf_summary}. Handled safely to prevent numerical overflow.")

    if const_cols:
        dq_points.append(f"Constant columns with zero sample variance: {', '.join(const_cols)}.")

    sections.append({
        "title": "1. Data Quality and Hygiene Findings",
        "content": " ".join(dq_points)
    })

    # 2. Distributional Characteristics
    dist_points = []
    skewed_cols = []
    heavy_tail_cols = []

    if not desc_stats.empty:
        for col in selected_cols:
            if col in desc_stats.columns:
                skew_val = desc_stats.loc["Skewness (Sample)", col] if "Skewness (Sample)" in desc_stats.index else np.nan
                kurt_val = desc_stats.loc["Kurtosis (Fisher's Excess)", col] if "Kurtosis (Fisher's Excess)" in desc_stats.index else np.nan

                if pd.notna(skew_val) and abs(skew_val) >= 1.0:
                    skewed_cols.append(f"{col} (skewness = {skew_val:.2f})")
                if pd.notna(kurt_val) and kurt_val > 1.0:
                    heavy_tail_cols.append(f"{col} (excess kurtosis = {kurt_val:.2f})")

        if skewed_cols:
            dist_points.append(
                f"Substantial distributional asymmetry was detected in {', '.join(skewed_cols)}. "
                "Because parametric methods assume symmetry, severe skewness can cause the standard Z-score rule "
                "to yield asymmetric or misleading outlier flags."
            )
        else:
            dist_points.append("Selected numerical variables exhibit approximately symmetric distributions.")

        if heavy_tail_cols:
            dist_points.append(
                f"Leptokurtic (heavy-tailed) properties observed in {', '.join(heavy_tail_cols)}. "
                "Heavy tails naturally produce extreme realizations that are legitimate sample values rather than measurement errors."
            )

    sections.append({
        "title": "2. Distributional Behavior and Shape Analysis",
        "content": " ".join(dist_points) if dist_points else "Distribution shape assessment complete."
    })

    # 3. Anomaly Detection & Method Agreement
    det_points = []
    cat_counts = outlier_summary.get("category_counts", {})
    both_cnt = cat_counts.get(CATEGORY_BOTH, 0)
    iqr_only_cnt = cat_counts.get(CATEGORY_IQR_ONLY, 0)
    z_only_cnt = cat_counts.get(CATEGORY_ZSCORE_ONLY, 0)
    neither_cnt = cat_counts.get(CATEGORY_NEITHER, 0)
    total_eval = both_cnt + iqr_only_cnt + z_only_cnt + neither_cnt

    det_points.append(
        f"Across {total_eval} analyzed observation-variable pairs: {both_cnt} were flagged by both methods, "
        f"{iqr_only_cnt} by IQR only, {z_only_cnt} by Z-score only, and {neither_cnt} by neither method."
    )

    if both_cnt > 0:
        det_points.append(
            f"The {both_cnt} points flagged by both methods exhibit high statistical divergence across both "
            "parametric (mean/variance) and non-parametric (interquartile) criteria, warranting prioritized audit review."
        )

    if iqr_only_cnt > 0 or z_only_cnt > 0:
        det_points.append(
            f"Method divergence ({iqr_only_cnt} IQR-only, {z_only_cnt} Z-score only) illustrates how detection rules "
            "depend on their statistical assumptions. Tukey's rule uses resistant quartiles, while Z-scores are influenced "
            "by sample standard deviation inflation (masking effect) from extreme points."
        )

    sections.append({
        "title": "3. Outlier Detection Synthesis and Methodological Concordance",
        "content": " ".join(det_points)
    })

    # 4. Hypothesis Testing Insights
    hyp_points = []
    for var, norm_res in normality_tests.items():
        if norm_res.get("is_valid", False):
            p_val = norm_res.get("p_value", 1.0)
            alpha = norm_res.get("alpha", 0.05)
            if p_val < alpha:
                hyp_points.append(
                    f"For variable '{var}', Shapiro-Wilk rejected normality (W={norm_res.get('statistic', 0.0):.4f}, p={p_val:.4e} < {alpha}). "
                    "This formal rejection confirms that normality-dependent outlier tests (such as Z-score and Grubbs') must be interpreted with caution."
                )
            else:
                hyp_points.append(
                    f"For variable '{var}', Shapiro-Wilk failed to reject normality (W={norm_res.get('statistic', 0.0):.4f}, p={p_val:.4f}). "
                    "Data is distributionally consistent with a Gaussian process, though this does not prove strict normality."
                )

    for var, g_res in grubbs_tests.items():
        if g_res.get("is_valid", False):
            if g_res.get("reject_null", False):
                hyp_points.append(
                    f"Grubbs' single-outlier test flagged the most extreme point in '{var}' "
                    f"(G={g_res.get('statistic_g', 0.0):.4f} > G_crit={g_res.get('critical_value', 0.0):.4f}, value={g_res.get('suspect_value', 0.0):.2f})."
                )

    if hyp_points:
        sections.append({
            "title": "4. Formal Hypothesis Testing Outcomes",
            "content": " ".join(hyp_points)
        })

    # 5. Methodological Limitations and Recommended Next Steps
    rec_points = [
        "1. Contextual Verification: Consult source domain records, timestamps, and data ingestion pipelines for flagged records.",
        "2. Biological / Physical Feasibility: Check whether flagged values represent impossible measurements (e.g., negative ages, impossible ratings) or legitimate high-volume transactions.",
        "3. Avoid Blind Truncation: Do not automatically discard flagged records; removing legitimate extreme points introduces sample truncation bias and artificially depresses variance.",
        "4. Independent Audit: Confirming intentional fabrication requires digital forensics, audit logs, or corroborating external evidence beyond mathematical outlier thresholds."
    ]

    sections.append({
        "title": "5. Critical Assumptions, Limitations, and Recommended Verification Steps",
        "content": "\n".join(rec_points)
    })

    return {
        "disclaimer": STATISTICAL_DISCLAIMER,
        "sections": sections,
        "key_takeaway": (
            f"Statistical analysis identified {both_cnt} high-concordance anomalies across {len(selected_cols)} variables. "
            "These findings establish statistical rarity, not intentional fraud or data fabrication."
        )
    }
