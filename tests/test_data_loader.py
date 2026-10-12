"""
Automated unit tests for data loader and dataset preparation module.
Covers Cases 11, 12, 13, 19, 20.
"""

import io
import pytest
import pandas as pd
import numpy as np

from core.data_loader import load_csv, load_demo_dataset, prepare_dataset
from config.settings import RECORD_ID_COL, ORIGINAL_ROW_COL


def test_load_demo_dataset():
    """Verify demo dataset loads correctly and contains expected columns and rows."""
    df = load_demo_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) >= 70
    assert "transaction_id" in df.columns
    assert "purchase_amount" in df.columns


def test_load_csv_utf8_and_latin1():
    """Verify loading CSV with various valid encodings."""
    csv_text = "id,val,category\n1,10.5,alpha\n2,20.2,beta\n"
    # UTF-8
    b_utf8 = io.BytesIO(csv_text.encode("utf-8"))
    df1 = load_csv(b_utf8)
    assert len(df1) == 2
    assert list(df1.columns) == ["id", "val", "category"]

    # Latin-1
    csv_latin1 = "id,city,temp\n1,München,22.5\n2,Zürich,19.0\n"
    b_latin = io.BytesIO(csv_latin1.encode("latin-1"))
    df2 = load_csv(b_latin)
    assert len(df2) == 2
    assert "München" in df2["city"].values


def test_load_empty_csv():
    """Verify empty CSV raises informative ValueError (Case 12)."""
    b_empty = io.BytesIO(b"")
    with pytest.raises(ValueError, match="empty"):
        load_csv(b_empty)


def test_load_malformed_csv():
    """Verify malformed CSV with inconsistent row fields raises ValueError (Case 13)."""
    # Inconsistent number of columns
    malformed = "col1,col2,col3\n1,2,3\n4,5\n6,7,8,9,10\n"
    b_malformed = io.BytesIO(malformed.encode("utf-8"))
    with pytest.raises(ValueError):
        load_csv(b_malformed)


def test_prepare_dataset_preserves_original_and_adds_provenance():
    """Verify prepare_dataset leaves original DataFrame untouched and assigns stable IDs (Cases 19, 20)."""
    original_df = pd.DataFrame({
        "item": ["A", "B", "C"],
        "price": [10.0, 20.0, 30.0]
    })
    original_copy = original_df.copy(deep=True)

    working_df = prepare_dataset(original_df)

    # 1. Original DataFrame must not be altered in shape or columns (Case 19)
    assert RECORD_ID_COL not in original_df.columns
    assert ORIGINAL_ROW_COL not in original_df.columns
    pd.testing.assert_frame_equal(original_df, original_copy)

    # 2. Working DataFrame has internal provenance columns
    assert RECORD_ID_COL in working_df.columns
    assert ORIGINAL_ROW_COL in working_df.columns
    assert list(working_df[RECORD_ID_COL]) == [0, 1, 2]
    assert list(working_df[ORIGINAL_ROW_COL]) == [1, 2, 3]


def test_prepare_dataset_with_duplicates():
    """Verify duplicates receive unique stable record IDs (Case 11, 20)."""
    df = pd.DataFrame({
        "name": ["Alice", "Alice", "Bob"],
        "val": [10, 10, 25]
    })
    prepared = prepare_dataset(df)
    assert len(prepared[RECORD_ID_COL].unique()) == 3
    assert list(prepared[RECORD_ID_COL]) == [0, 1, 2]
    assert list(prepared[ORIGINAL_ROW_COL]) == [1, 2, 3]
