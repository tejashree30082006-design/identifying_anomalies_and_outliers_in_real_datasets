"""
Configuration settings and constants for the Statistical Data Anomaly Detector.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEMO_DATASET_PATH = DATA_DIR / "demo_dataset.csv"

# Project Metadata
APP_TITLE = "Can Statistics Detect Fake Data?"
APP_SUBTITLE = "Identifying Anomalies and Outliers in Real Datasets Using Statistical Methodology"
VERSION = "1.0.0"
ACADEMIC_CONTEXT = "College Statistics Project Demonstration & Evaluation"

# Upload and Processing Limits
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = [".csv"]
SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

# Internal Row Tracking Columns
RECORD_ID_COL = "_record_id"
ORIGINAL_ROW_COL = "_original_row_number"

# Statistical Defaults
DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_ALPHA = 0.05

# Sample Size Constraints
SHAPIRO_MIN_SAMPLE = 3
SHAPIRO_MAX_OFFICIAL_SAMPLE = 5000
GRUBBS_MIN_SAMPLE = 3

# Quantile convention
# 'linear' is standard in pandas/numpy, corresponding to type 7 in R/Hyndman & Fan (1996)
QUANTILE_METHOD = "linear"

# Visual Theme & Colors (Accessible Academic Palette)
PALETTE = {
    "clean": "#2b5c8f",          # Slate Navy
    "iqr_only": "#f39c12",       # Amber / Orange
    "zscore_only": "#8e44ad",    # Purple
    "both_methods": "#c0392b",   # Crimson
    "neither": "#27ae60",        # Forest Green
    "fences": "#e74c3c",         # Red for fence lines
    "normal_fit": "#2980b9",     # Blue for fitted PDF
    "highlight": "#d35400",      # Rust Orange
    "grid": "#ecf0f1",           # Light Gray
    "text": "#2c3e50"            # Dark Charcoal
}

# Standard Disclaimers & Method Definitions
STATISTICAL_DISCLAIMER = (
    "CRITICAL METHODOLOGICAL NOTICE: Statistical outlier detection identifies observations "
    "that deviate substantially from an assumed distributional pattern or the bulk of the data. "
    "It cannot and does not prove deliberate fabrication or fraud. An outlier may reflect a valid rare event, "
    "a measurement error, a data entry typo, heavy-tailed population phenomena, or a model misspecification. "
    "Statistical flags must always be combined with domain context, audit trails, and independent validation."
)

METHOD_DESCRIPTIONS = {
    "iqr": (
        "Tukey's Interquartile Range (IQR) Rule: A non-parametric method resistant to extreme values. "
        "Fences are defined as Q1 - k*IQR and Q3 + k*IQR (default k=1.5). Observations outside this fence "
        "are flagged. Does not assume normality, but can struggle with zero-IQR discrete distributions."
    ),
    "zscore": (
        "Standard Z-Score Method: Parametric metric expressing how many sample standard deviations a value "
        "lies from the sample mean. Observations with |Z| > threshold (default 3.0) are flagged. "
        "Assumes approximate normality and is sensitive to masking effects from extreme outliers inflating the sample variance."
    ),
    "shapiro": (
        "Shapiro-Wilk Normality Test: Evaluates H0: data is normally distributed vs H1: data is not normally distributed. "
        "Valid for sample sizes between 3 and 5,000 observations. Rejecting H0 means evidence against normality; "
        "failing to reject H0 does not prove the population is normal."
    ),
    "grubbs": (
        "Grubbs' Test for Outliers (Maximum Normed Residual): Tests H0: no outliers in a normal sample vs "
        "H1: exactly one outlier exists (at either minimum or maximum). Assumes strict underlying normality "
        "and is formally specified for a single suspect observation."
    )
}

# Detection Categories
CATEGORY_BOTH = "Flagged by Both Methods"
CATEGORY_IQR_ONLY = "Flagged by IQR Only"
CATEGORY_ZSCORE_ONLY = "Flagged by Z-Score Only"
CATEGORY_NEITHER = "Flagged by Neither Method"
