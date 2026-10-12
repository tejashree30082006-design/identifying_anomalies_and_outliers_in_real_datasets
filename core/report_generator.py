"""
Analytical report generator module.
Builds an exhaustive, transparent, academically formatted Markdown audit dossier
from actual calculated results, settings, and statistical tests.
"""

from typing import Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

from config.settings import (
    APP_TITLE,
    VERSION,
    STATISTICAL_DISCLAIMER,
    METHOD_DESCRIPTIONS,
    CATEGORY_BOTH,
    CATEGORY_IQR_ONLY,
    CATEGORY_ZSCORE_ONLY,
    CATEGORY_NEITHER
)
from core.interpretation import generate_interpretation


def generate_markdown_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate a full Markdown analytical report from calculated analysis results.
    Never uses hardcoded numbers or speculative claims.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = analysis_results.get("dataset_name", "User Dataset")
    selected_cols = analysis_results.get("selected_columns", [])
    quality = analysis_results.get("quality", {})
    desc_stats = analysis_results.get("desc_stats", pd.DataFrame())
    outlier_summary = analysis_results.get("outlier_summary", {})
    normality_tests = analysis_results.get("normality_tests", {})
    grubbs_tests = analysis_results.get("grubbs_tests", {})
    z_thresh = analysis_results.get("z_threshold", 3.0)
    iqr_mult = analysis_results.get("iqr_multiplier", 1.5)
    alpha = analysis_results.get("alpha", 0.05)

    interp_data = generate_interpretation(analysis_results)

    lines = []
    lines.append(f"# Statistical Data Anomaly & Outlier Audit Report")
    lines.append(f"**Project:** {APP_TITLE} (Version {VERSION})  ")
    lines.append(f"**Generated:** {now_str}  ")
    lines.append(f"**Dataset Source:** `{dataset_name}`  ")
    lines.append(f"**Selected Variables:** `{', '.join(selected_cols)}`  ")
    lines.append("\n---\n")

    # Executive Disclaimer
    lines.append("## Methodological Notice & Scientific Disclaimer")
    lines.append(f"> **IMPORTANT:** {STATISTICAL_DISCLAIMER}\n")

    # 1. Dataset Dimensions & Quality
    lines.append("## 1. Dataset Overview & Data Quality Audit")
    lines.append(f"- **Total Records:** {quality.get('total_rows', 0)}")
    lines.append(f"- **Total Columns:** {quality.get('total_columns', 0)}")
    lines.append(f"- **Identified Numerical Columns:** `{', '.join(quality.get('numeric_columns', []))}`")
    lines.append(f"- **Identified Categorical Columns:** `{', '.join(quality.get('categorical_columns', []))}`")

    missing_cells = quality.get("total_missing_cells", 0)
    lines.append(f"- **Total Missing Cells:** {missing_cells}")

    dup_info = quality.get("duplicate_summary", {})
    lines.append(f"- **Duplicate Rows:** {dup_info.get('duplicate_count', 0)} ({dup_info.get('duplicate_percentage', 0.0)}%)")

    inf_counts = quality.get("infinite_counts", {})
    if inf_counts:
        lines.append(f"- **Infinite Values Detected:** {inf_counts}")
    else:
        lines.append("- **Infinite Values:** None detected.")

    const_cols = quality.get("constant_columns", [])
    if const_cols:
        lines.append(f"- **Zero-Variance / Constant Columns:** `{', '.join(const_cols)}`")
    lines.append("\n")

    # 2. Descriptive Statistics Table
    lines.append("## 2. Descriptive Statistics Table")
    if not desc_stats.empty:
        # Build Markdown table natively without external tabulate dependency
        headers = ["Metric"] + [str(c) for c in desc_stats.columns]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        lines.append(header_row)
        lines.append(sep_row)
        for idx in desc_stats.index:
            row_vals = [str(idx)]
            for col in desc_stats.columns:
                v = desc_stats.loc[idx, col]
                if isinstance(v, (int, float, np.floating, np.integer)) and pd.notna(v) and not np.isinf(v):
                    row_vals.append(f"{v:.4f}")
                else:
                    row_vals.append(str(v))
            lines.append("| " + " | ".join(row_vals) + " |")
    else:
        lines.append("*No numerical columns analyzed.*")
    lines.append("\n")

    # 3. Outlier Detection Parameters & Tallies
    lines.append("## 3. Statistical Outlier Detection Methodology & Results")
    lines.append(f"- **Tukey IQR Multiplier:** `k = {iqr_mult}` (Fences: Q1 - {iqr_mult}×IQR to Q3 + {iqr_mult}×IQR)")
    lines.append(f"- **Z-Score Threshold:** `|Z| > {z_thresh}` (using sample standard deviation with ddof=1)")
    lines.append("\n### Detection Summary by Mutually Exclusive Categories\n")

    cat_counts = outlier_summary.get("category_counts", {})
    lines.append("| Classification Category | Analyzed Pairs | Description |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **{CATEGORY_BOTH}** | {cat_counts.get(CATEGORY_BOTH, 0)} | High divergence across both parametric and non-parametric fences |")
    lines.append(f"| **{CATEGORY_IQR_ONLY}** | {cat_counts.get(CATEGORY_IQR_ONLY, 0)} | Flagged by resistant quartiles; within standard deviation threshold |")
    lines.append(f"| **{CATEGORY_ZSCORE_ONLY}** | {cat_counts.get(CATEGORY_ZSCORE_ONLY, 0)} | Flagged by sample Z-score; within Tukey fences |")
    lines.append(f"| **{CATEGORY_NEITHER}** | {cat_counts.get(CATEGORY_NEITHER, 0)} | Within standard expected variation under both rules |")
    lines.append("\n")

    uniq_summary = outlier_summary.get("unique_records_summary", {})
    if uniq_summary:
        lines.append("### Unique Source Records Audit")
        lines.append(f"- **Total Unique Source Records:** {uniq_summary.get('total_unique_records', 0)}")
        lines.append(f"- **Flagged Unique Records (Any Variable):** {uniq_summary.get('flagged_unique_records', 0)} ({uniq_summary.get('flagged_record_percentage', 0.0)}%)")
        lines.append(f"- **Completely Clean Unique Records:** {uniq_summary.get('unflagged_unique_records', 0)}")
        lines.append("\n")

    # 4. Hypothesis Testing Section
    lines.append("## 4. Formal Hypothesis Testing")
    lines.append(f"**Significance Level (α):** `{alpha}`\n")

    lines.append("### A. Shapiro-Wilk Normality Tests")
    if normality_tests:
        for var, n_res in normality_tests.items():
            lines.append(f"#### Variable: `{var}`")
            if n_res.get("is_valid", False):
                lines.append(f"- **W-Statistic:** `{n_res.get('statistic', 0.0):.4f}`")
                lines.append(f"- **P-Value:** `{n_res.get('p_value', 0.0):.4e}`")
                lines.append(f"- **Decision:** **{n_res.get('decision', 'N/A')}**")
                lines.append(f"- **Interpretation:** {n_res.get('interpretation', '')}")
                if n_res.get("methodological_caveats"):
                    lines.append(f"- **Caveat:** {n_res.get('methodological_caveats')}")
            else:
                lines.append(f"- *{n_res.get('interpretation', 'Test could not be computed.')}*")
            lines.append("")
    else:
        lines.append("*Normality tests not computed.*")

    lines.append("### B. Grubbs' Single-Outlier Tests")
    if grubbs_tests:
        for var, g_res in grubbs_tests.items():
            lines.append(f"#### Variable: `{var}`")
            if g_res.get("is_valid", False):
                lines.append(f"- **Grubbs' G-Statistic:** `{g_res.get('statistic_g', 0.0):.4f}`")
                lines.append(f"- **Critical Value (G_crit):** `{g_res.get('critical_value', 0.0):.4f}`")
                lines.append(f"- **Suspect Value Tested:** `{g_res.get('suspect_value', 0.0):.2f}` ({g_res.get('suspect_direction', '')})")
                lines.append(f"- **Decision:** **{g_res.get('decision', 'N/A')}**")
                lines.append(f"- **Interpretation:** {g_res.get('interpretation', '')}")
            else:
                lines.append(f"- *{g_res.get('interpretation', 'Grubbs test could not be computed.')}*")
            lines.append("")
    else:
        lines.append("*Grubbs' tests not computed.*")
    lines.append("\n")

    # 5. Methodological Interpretation & Next Steps
    lines.append("## 5. Analytical Interpretation & Audit Recommendations")
    for sec in interp_data.get("sections", []):
        lines.append(f"### {sec['title']}")
        lines.append(f"{sec['content']}\n")

    lines.append("## 6. Statistical Method Definitions Reference")
    for method, desc in METHOD_DESCRIPTIONS.items():
        lines.append(f"- **{method.upper()}**: {desc}")
    lines.append("\n")

    return "\n".join(lines)
