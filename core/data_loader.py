"""
Data loading and dataset preparation module.
Handles multi-encoding CSV imports, demonstration dataset loading,
and safe preparation with stable record identifiers without altering original records.
"""

from typing import Union, Optional
from pathlib import Path
import io
import pandas as pd

from config.settings import (
    DEMO_DATASET_PATH,
    SUPPORTED_ENCODINGS,
    RECORD_ID_COL,
    ORIGINAL_ROW_COL
)
from utils.validation import validate_file_size, validate_file_extension


def load_csv(uploaded_file: Union[str, Path, io.BytesIO, object]) -> pd.DataFrame:
    """
    Load a CSV file with automatic fallback across standard encodings.
    Handles file paths, BytesIO, and Streamlit UploadedFile objects.
    
    Raises:
        ValueError: If file cannot be read, decoded, or parsed.
    """
    if uploaded_file is None:
        raise ValueError("No file provided for upload.")

    # Validate file size if object provides size attributes
    is_valid_size, size_err = validate_file_size(uploaded_file)
    if not is_valid_size:
        raise ValueError(size_err)

    # Check filename extension if file is a path or named object
    if isinstance(uploaded_file, (str, Path)):
        is_valid_ext, ext_err = validate_file_extension(str(uploaded_file))
        if not is_valid_ext:
            raise ValueError(ext_err)
    elif hasattr(uploaded_file, "name"):
        is_valid_ext, ext_err = validate_file_extension(uploaded_file.name)
        if not is_valid_ext:
            raise ValueError(ext_err)

    # Read binary bytes once to allow multiple encoding attempts
    raw_bytes: bytes
    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            raw_bytes = f.read()
    elif hasattr(uploaded_file, "read"):
        # Reset position if seekable
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    elif hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    else:
        raise ValueError("Unsupported file source format.")

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValueError("The provided CSV file is empty (0 bytes).")

    # Try each encoding sequentially
    last_error: Optional[Exception] = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            buffer = io.BytesIO(raw_bytes)
            df = pd.read_csv(buffer, encoding=encoding)
            if df.empty and len(df.columns) == 0:
                raise ValueError("Parsed file contains no headers or data rows.")

            # If CSV parsed as a single column but contains delimiter (e.g., fully quoted rows)
            if len(df.columns) == 1 and len(df) > 0:
                col_name = str(df.columns[0])
                for sep in [",", ";", "\t", "|"]:
                    if sep in col_name:
                        split_cols = [c.strip().strip('"').strip("'") for c in col_name.split(sep)]
                        split_df = df.iloc[:, 0].astype(str).str.split(sep, expand=True)
                        if split_df.shape[1] == len(split_cols):
                            split_df.columns = split_cols
                            for c in split_df.columns:
                                try:
                                    split_df[c] = pd.to_numeric(split_df[c])
                                except Exception:
                                    pass
                            df = split_df
                            break

            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            last_error = e
            # If it's a parse error other than decoding, retry with next or raise
            continue

    err_msg = (
        f"Failed to decode or parse CSV using supported encodings ({', '.join(SUPPORTED_ENCODINGS)}). "
        f"Detail: {last_error if last_error else 'Unknown decoding error.'}"
    )
    raise ValueError(err_msg)


def load_demo_dataset() -> pd.DataFrame:
    """Load the built-in local synthetic demonstration dataset."""
    if not DEMO_DATASET_PATH.is_file():
        raise FileNotFoundError(f"Demo dataset not found at {DEMO_DATASET_PATH}")
    return pd.read_csv(DEMO_DATASET_PATH, encoding="utf-8")


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a working dataset copy with stable row identifiers.
    Preserves original row order and keeps all data untouched.
    Adds internal metadata columns:
      - _record_id: Unique integer index (0, 1, 2, ...)
      - _original_row_number: 1-indexed source row number (1, 2, 3, ...)
      
    Never deletes missing values, duplicates, or extreme values.
    """
    if df is None:
        raise ValueError("Cannot prepare a None dataset.")

    # Create an explicit deep copy to ensure complete isolation
    working_df = df.copy(deep=True)

    # Insert stable identifier columns
    working_df[RECORD_ID_COL] = list(range(len(working_df)))
    working_df[ORIGINAL_ROW_COL] = [i + 1 for i in range(len(working_df))]

    # Keep original order intact
    return working_df
