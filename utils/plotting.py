"""
Centralized plotting module.
Generates polished, labeled, reproducible Matplotlib/Seaborn figures
with accessible palettes, explicit fence/threshold lines, and outlier highlights.
"""

from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

from config.settings import PALETTE
from utils.validation import sanitize_numeric_series


def setup_plot_style():
    """Apply consistent, professional styling parameters across figures."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e5e5"


def plot_histogram_with_fit(
    series: pd.Series,
    var_name: str,
    iqr_lower: Optional[float] = None,
    iqr_upper: Optional[float] = None,
    show_normal_fit: bool = True
) -> plt.Figure:
    """
    Plot empirical histogram with optional estimated normal PDF overlay
    and Tukey IQR fences.
    """
    setup_plot_style()
    valid_s = sanitize_numeric_series(series)
    vals = valid_s.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    if len(vals) == 0:
        ax.text(0.5, 0.5, "No valid finite data for histogram", ha="center", va="center")
        ax.set_title(f"Histogram: {var_name}")
        fig.tight_layout()
        return fig

    # Base histogram
    num_bins = min(35, max(8, int(np.sqrt(len(vals)))))
    count, bins, patches = ax.hist(
        vals,
        bins=num_bins,
        density=show_normal_fit and len(vals) > 2,
        color=PALETTE["clean"],
        alpha=0.65,
        edgecolor="#ffffff",
        label="Empirical Distribution"
    )

    # Optional fitted normal curve (labeled as estimate)
    if show_normal_fit and len(vals) >= 3:
        mu = float(np.mean(vals))
        sigma = float(np.std(vals, ddof=1))
        if sigma > 0:
            x_grid = np.linspace(min(vals), max(vals), 200)
            pdf = stats.norm.pdf(x_grid, mu, sigma)
            ax.plot(
                x_grid,
                pdf,
                color=PALETTE["normal_fit"],
                linewidth=2.2,
                linestyle="--",
                label=f"Fitted Normal PDF (μ={mu:.1f}, σ={sigma:.1f})"
            )

    # Draw Tukey fences if provided
    if iqr_lower is not None and not np.isnan(iqr_lower):
        ax.axvline(
            iqr_lower,
            color=PALETTE["fences"],
            linestyle=":",
            linewidth=1.8,
            label=f"Lower Fence ({iqr_lower:.2f})"
        )
    if iqr_upper is not None and not np.isnan(iqr_upper):
        ax.axvline(
            iqr_upper,
            color=PALETTE["fences"],
            linestyle=":",
            linewidth=1.8,
            label=f"Upper Fence ({iqr_upper:.2f})"
        )

    ax.set_title(f"Empirical Histogram & Fences: {var_name}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(f"{var_name} (Values)", fontsize=10)
    ax.set_ylabel("Density" if (show_normal_fit and len(vals) > 2) else "Frequency", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    return fig


def plot_boxplot(
    series: pd.Series,
    var_name: str,
    iqr_multiplier: float = 1.5,
    flagged_values: Optional[List[float]] = None
) -> plt.Figure:
    """
    Generate horizontal boxplot highlighting Tukey fences and detected outliers.
    """
    setup_plot_style()
    valid_s = sanitize_numeric_series(series)
    vals = valid_s.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=100)

    if len(vals) == 0:
        ax.text(0.5, 0.5, "No valid data for boxplot", ha="center", va="center")
        ax.set_title(f"Boxplot: {var_name}")
        fig.tight_layout()
        return fig

    # Boxplot
    bp = ax.boxplot(
        vals,
        vert=False,
        patch_artist=True,
        whis=iqr_multiplier,
        boxprops=dict(facecolor=PALETTE["clean"], color="#1e3f66", alpha=0.6),
        medianprops=dict(color="#d35400", linewidth=2.0),
        whiskerprops=dict(color="#1e3f66", linewidth=1.2),
        capprops=dict(color="#1e3f66", linewidth=1.2),
        flierprops=dict(marker="o", markerfacecolor=PALETTE["highlight"], markeredgecolor="#ffffff", markersize=6)
    )

    # Highlight flagged values if explicitly passed
    if flagged_values:
        clean_flags = [v for v in flagged_values if np.isfinite(v)]
        if clean_flags:
            ax.scatter(
                clean_flags,
                [1.0] * len(clean_flags),
                color=PALETTE["both_methods"],
                s=65,
                zorder=5,
                edgecolors="#ffffff",
                label=f"Flagged Outliers (N={len(clean_flags)})"
            )
            ax.legend(loc="upper right", fontsize=9)

    ax.set_title(f"Tukey Boxplot (k={iqr_multiplier}): {var_name}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(f"{var_name} (Values)", fontsize=10)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_qq(series: pd.Series, var_name: str) -> plt.Figure:
    """
    Plot Quantile-Quantile (Q-Q) plot against theoretical standard normal distribution.
    """
    setup_plot_style()
    valid_s = sanitize_numeric_series(series)
    vals = valid_s.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

    if len(vals) < 3:
        ax.text(0.5, 0.5, "Insufficient sample size for Q-Q plot (N < 3)", ha="center", va="center")
        ax.set_title(f"Normal Q-Q Plot: {var_name}")
        fig.tight_layout()
        return fig

    # Generate probability plot
    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
    
    ax.scatter(osm, osr, color=PALETTE["clean"], alpha=0.75, edgecolors="#ffffff", s=40, label="Sample Quantiles")
    # Reference line
    x_line = np.array([min(osm), max(osm)])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=PALETTE["highlight"], linewidth=2.0, label=f"Theoretical Reference (R²={r**2:.3f})")

    ax.set_title(f"Normal Q-Q Plot: {var_name}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Theoretical Standard Normal Quantiles", fontsize=10)
    ax.set_ylabel(f"Sample Quantiles ({var_name})", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_zscores(
    series: pd.Series,
    var_name: str,
    threshold: float = 3.0,
    zscore_series: Optional[pd.Series] = None
) -> plt.Figure:
    """
    Plot scatter plot of signed Z-scores against row index with ±threshold boundary lines.
    """
    setup_plot_style()
    s_numeric = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(s_numeric)
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    if finite_mask.sum() < 2:
        ax.text(0.5, 0.5, "Insufficient data for Z-score plot", ha="center", va="center")
        ax.set_title(f"Z-Score Deviations: {var_name}")
        fig.tight_layout()
        return fig

    # Calculate Z-scores if not supplied
    if zscore_series is None or zscore_series.empty:
        vals = s_numeric[finite_mask].to_numpy(dtype=float)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))
        if std_val > 0:
            z_vals = (s_numeric - mean_val) / std_val
        else:
            z_vals = pd.Series(0.0, index=series.index)
    else:
        z_vals = zscore_series

    indices = list(range(1, len(z_vals) + 1))
    z_clean = z_vals.fillna(0.0).to_numpy(dtype=float)

    # Classify colors
    colors = [PALETTE["both_methods"] if abs(z) > threshold else PALETTE["clean"] for z in z_clean]

    ax.scatter(indices, z_clean, c=colors, alpha=0.8, s=45, edgecolors="#ffffff", zorder=3)
    ax.axhline(0, color="#888888", linestyle="-", linewidth=1.0)
    ax.axhline(threshold, color=PALETTE["fences"], linestyle="--", linewidth=1.5, label=f"+{threshold:.1f}σ Threshold")
    ax.axhline(-threshold, color=PALETTE["fences"], linestyle="--", linewidth=1.5, label=f"-{threshold:.1f}σ Threshold")

    ax.set_title(f"Signed Z-Score Plot: {var_name} (Threshold = ±{threshold:.1f})", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Source Record Row Number", fontsize=10)
    ax.set_ylabel("Signed Z-Score (Standard Deviations)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_method_comparison_counts(category_counts: Dict[str, int]) -> plt.Figure:
    """
    Generate horizontal bar chart comparing observation counts
    across the four mutually exclusive detection categories.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=100)

    cats = list(category_counts.keys())
    counts = list(category_counts.values())

    color_map = [
        PALETTE["both_methods"],   # Both
        PALETTE["iqr_only"],       # IQR only
        PALETTE["zscore_only"],    # Z-score only
        PALETTE["neither"]         # Neither
    ]

    bars = ax.barh(cats, counts, color=color_map[:len(cats)], alpha=0.85, edgecolor="#ffffff")

    # Add numeric labels to bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(0.2, max(counts) * 0.01 if counts else 0.5), bar.get_y() + bar.get_height() / 2,
                f"{int(w)}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_title("Method Concordance: Mutually Exclusive Categories", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Count of Analyzed Observations / Pairs", fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_missing_values(missing_summary_df: pd.DataFrame) -> plt.Figure:
    """
    Plot bar chart of missing value percentages per column.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    if missing_summary_df.empty or "Column" not in missing_summary_df.columns:
        ax.text(0.5, 0.5, "No missing value data available", ha="center", va="center")
        fig.tight_layout()
        return fig

    cols = missing_summary_df["Column"].tolist()
    pcts = missing_summary_df["Missing Percentage"].tolist()

    bars = ax.bar(cols, pcts, color=PALETTE["clean"], alpha=0.75, edgecolor="#ffffff")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_title("Data Quality Audit: Missing Values by Column (%)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Missing Percentage (%)", fontsize=10)
    ax.set_ylim(0, max(10, max(pcts) + 5) if pcts else 10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig
