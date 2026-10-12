"""
Data validation and sanity-checking utilities.
Ensures data integrity, file safety, and numeric compliance without modifying original data.
"""

from typing import List, Tuple, Optional
import io
import pandas as pd
import numpy as np

from config.settings import (
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    SHAPIRO_MIN_SAMPLE,
    SHAPIRO_MAX_OFFICIAL_SAMPLE,
    GRUBBS_MIN_SAMPLE
)


def validate_file_extension(filename: str) -> Tuple[bool, Optional[str]]:
    """Verify that uploaded file has an allowed extension."""
    if not filename:
        return False, "No filename provided."
    lower_name = filename.lower()
    if any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return True, None
    return False, f"Unsupported file extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"


def validate_file_size(file_obj) -> Tuple[bool, Optional[str]]:
    """Check that uploaded file does not exceed maximum allowable size."""
    try:
        if hasattr(file_obj, "size"):
            size = file_obj.size
        elif hasattr(file_obj, "getbuffer"):
            size = len(file_obj.getbuffer())
        elif hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
            curr = file_obj.tell()
            file_obj.seek(0, io.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(curr)
        else:
            return True, None  # Cannot determine size reliably, proceed with caution
        
        if size > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            return False, f"File size ({size / (1024 * 1024):.2f} MB) exceeds maximum limit of {max_mb:.0f} MB."
        return True, None
    except Exception as e:
        return True, None


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns that are numeric in type, excluding internal metadata columns."""
    if df is None or df.empty:
        return []
    numeric_cols = []
    for col in df.columns:
        if str(col).startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
    return numeric_cols


def sanitize_numeric_series(series: pd.Series) -> pd.Series:
    """
    Extract valid finite numeric observations from a series.
    Returns a copy containing only finite numbers (excluding NaN, +inf, -inf).
    Original series is unmodified.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    # Convert safely to numeric if object dtype
    s_num = pd.to_numeric(series, errors="coerce")
    # Mask finite values
    valid_mask = np.isfinite(s_num)
    return s_num[valid_mask].copy()


def check_sample_size_for_normality(n: int) -> Tuple[bool, str]:
    """
    Check if sample size is valid for Shapiro-Wilk normality test.
    Returns (is_valid, message).
    """
    if n < SHAPIRO_MIN_SAMPLE:
        return False, f"Sample size (N={n}) is too small for Shapiro-Wilk test (minimum N={SHAPIRO_MIN_SAMPLE} required)."
    if n > SHAPIRO_MAX_OFFICIAL_SAMPLE:
        return True, (
            f"Sample size (N={n}) is large (> {SHAPIRO_MAX_OFFICIAL_SAMPLE}). "
            "Shapiro-Wilk test has very high statistical power with large samples, meaning even tiny, "
            "practically trivial departures from normality will produce very small p-values."
        )
    return True, f"Sample size (N={n}) is within valid range ({SHAPIRO_MIN_SAMPLE} to {SHAPIRO_MAX_OFFICIAL_SAMPLE})."


def check_sample_size_for_grubbs(n: int) -> Tuple[bool, str]:
    """
    Check if sample size is valid for Grubbs' outlier test.
    Returns (is_valid, message).
    """
    if n < GRUBBS_MIN_SAMPLE:
        return False, f"Sample size (N={n}) is too small for Grubbs' test (minimum N={GRUBBS_MIN_SAMPLE} required)."
    return True, f"Sample size (N={n}) is sufficient for single-outlier Grubbs test."
