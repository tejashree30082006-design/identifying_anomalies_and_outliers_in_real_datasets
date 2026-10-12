"""
Automated unit tests verifying the user-provided CSV dynamic analysis workflow.
Tests two distinct CSV files with different column schemas, row counts, and numerical values.
Confirms statistics and flagged records adapt dynamically to each dataset.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from core.data_loader import load_csv, prepare_dataset
from core.data_quality import analyze_data_quality
from core.descriptive_stats import calculate_descriptive_statistics
from core.outlier_detection import detect_all_outliers
from core.result_comparison import compare_detection_methods
from core.report_generator import generate_markdown_report
from utils.validation import get_numeric_columns
from config.settings import RECORD_ID_COL, ORIGINAL_ROW_COL


TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
HOUSING_CSV_PATH = TEST_DATA_DIR / "housing_sample.csv"
SENSOR_CSV_PATH = TEST_DATA_DIR / "sensor_telemetry.csv"


def test_first_user_csv_housing_dataset():
    """Verify complete dynamic analysis pipeline on first user CSV (Housing dataset)."""
    assert HOUSING_CSV_PATH.is_file()

    # 1. Load and Prepare
    raw_df = load_csv(HOUSING_CSV_PATH)
    assert len(raw_df) == 40
    assert "sale_price_thousands" in raw_df.columns
    assert "square_feet" in raw_df.columns
    assert "temperature_c" not in raw_df.columns

    working_df = prepare_dataset(raw_df)
    assert RECORD_ID_COL in working_df.columns
    assert ORIGINAL_ROW_COL in working_df.columns

    # 2. Quality Audit
    quality = analyze_data_quality(working_df)
    assert quality["total_rows"] == 40
    assert quality["total_columns"] == 5
    num_cols = get_numeric_columns(working_df)
    assert set(num_cols) == {"square_feet", "lot_size_acres", "sale_price_thousands", "building_age"}

    # 3. Descriptive Statistics
    stats_df = calculate_descriptive_statistics(working_df, num_cols)
    assert "sale_price_thousands" in stats_df.columns
    # Extreme price of 9500 inflates the mean significantly above median
    mean_price = stats_df.loc["Mean", "sale_price_thousands"]
    median_price = stats_df.loc["Median", "sale_price_thousands"]
    assert mean_price > median_price

    # 4. Outlier Detection
    det_bundle = detect_all_outliers(working_df, num_cols, z_threshold=3.0, iqr_multiplier=1.5)
    comp_results = compare_detection_methods(det_bundle["combined_df"], det_bundle["combined_df"])
    comp_df = comp_results["comparison_df"]

    # Verify injected outlier at row 6 (index 5)
    flagged_price = comp_df[(comp_df["variable"] == "sale_price_thousands") & (comp_df["iqr_flag"] == True)]
    assert len(flagged_price) >= 1
    assert 9500.0 in flagged_price["original_value"].values
    assert 6 in flagged_price[ORIGINAL_ROW_COL].values

    # 5. Report Generation
    report = generate_markdown_report({
        "dataset_name": "housing_sample.csv",
        "selected_columns": num_cols,
        "quality": quality,
        "desc_stats": stats_df,
        "outlier_summary": comp_results,
        "normality_tests": {},
        "grubbs_tests": {}
    })
    assert "housing_sample.csv" in report
    assert "sale_price_thousands" in report


def test_second_user_csv_sensor_dataset():
    """Verify complete dynamic analysis pipeline on second user CSV (Sensor dataset)."""
    assert SENSOR_CSV_PATH.is_file()

    # 1. Load and Prepare
    raw_df = load_csv(SENSOR_CSV_PATH)
    assert len(raw_df) == 60
    assert "temperature_c" in raw_df.columns
    assert "pressure_kpa" in raw_df.columns
    assert "sale_price_thousands" not in raw_df.columns

    working_df = prepare_dataset(raw_df)

    # 2. Quality Audit
    quality = analyze_data_quality(working_df)
    assert quality["total_rows"] == 60
    assert quality["total_columns"] == 5
    num_cols = get_numeric_columns(working_df)
    assert set(num_cols) == {"temperature_c", "pressure_kpa", "vibration_hz", "voltage_v"}

    # 3. Descriptive Statistics
    stats_df = calculate_descriptive_statistics(working_df, num_cols)
    assert "temperature_c" in stats_df.columns
    # 850 C outlier pulls mean temperature
    mean_temp = stats_df.loc["Mean", "temperature_c"]
    median_temp = stats_df.loc["Median", "temperature_c"]
    assert mean_temp > median_temp

    # 4. Outlier Detection
    det_bundle = detect_all_outliers(working_df, num_cols, z_threshold=3.0, iqr_multiplier=1.5)
    comp_results = compare_detection_methods(det_bundle["combined_df"], det_bundle["combined_df"])
    comp_df = comp_results["comparison_df"]

    # Verify injected extreme temperature at row 13 (index 12)
    flagged_temp = comp_df[(comp_df["variable"] == "temperature_c") & (comp_df["iqr_flag"] == True)]
    assert len(flagged_temp) >= 1
    assert 850.0 in flagged_temp["original_value"].values
    assert 13 in flagged_temp[ORIGINAL_ROW_COL].values

    # Verify negative pressure outlier at row 35 (index 34)
    flagged_press = comp_df[(comp_df["variable"] == "pressure_kpa") & (comp_df["iqr_flag"] == True)]
    assert len(flagged_press) >= 1
    assert -120.0 in flagged_press["original_value"].values
    assert 35 in flagged_press[ORIGINAL_ROW_COL].values


def test_dynamic_variation_between_datasets():
    """Verify statistics and outlier flags change dynamically between distinct uploaded datasets."""
    raw1 = load_csv(HOUSING_CSV_PATH)
    raw2 = load_csv(SENSOR_CSV_PATH)

    prep1 = prepare_dataset(raw1)
    prep2 = prepare_dataset(raw2)

    cols1 = get_numeric_columns(prep1)
    cols2 = get_numeric_columns(prep2)

    # Columns must be completely distinct
    assert set(cols1).isdisjoint(set(cols2))

    stats1 = calculate_descriptive_statistics(prep1, cols1)
    stats2 = calculate_descriptive_statistics(prep2, cols2)

    # Metrics DataFrame must differ completely in columns and values
    assert list(stats1.columns) != list(stats2.columns)
    assert stats1.loc["Valid Observations", cols1[0]] == 40
    assert stats2.loc["Valid Observations", cols2[0]] == 60
