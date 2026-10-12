# Synthetic Demonstration Dataset Documentation

## Dataset Title
**Synthetic Retail Transaction and Customer Behavior Audit Dataset**

### Important Disclaimers
> **ATTENTION**: This dataset is **100% synthetic demonstration data** created strictly for educational and testing purposes. It does **not** represent actual customer, banking, retail, or credit transactions. All values, customer identifiers, and transaction records were mathematically simulated.

---

### Purpose
This dataset is designed to evaluate statistical data quality, descriptive metrics, distribution properties, Tukey IQR fences, Z-score detection thresholds, and hypothesis tests. It includes ordinary variation alongside deliberately introduced anomalies and data-hygiene challenges.

---

### Variable Dictionary

| Column Name | Type | Unit / Range | Typical Expected Behavior |
| :--- | :--- | :--- | :--- |
| `transaction_id` | Categorical (String) | Identifier | Unique transaction code (e.g., `TXN-1001`) |
| `customer_age` | Numerical (Integer) | Years (18 - 85) | Customer age at purchase time |
| `purchase_amount` | Numerical (Float) | USD ($10 - $250) | Total dollar amount spent in the transaction |
| `items_count` | Numerical (Integer) | Count (1 - 15) | Total quantity of merchandise items purchased |
| `session_duration_min` | Numerical (Float) | Minutes (2.0 - 60.0) | Browsing or checkout duration |
| `satisfaction_score` | Numerical (Float) | Rating (1.0 - 5.0) | Post-transaction survey rating |
| `store_channel` | Categorical (String) | Web / Mobile / In-Store | Sales channel |

---

### Ground-Truth Injection Log (Kept strictly outside detection logic)

The following deliberate anomalies were injected to evaluate specific detection components:

1. **Extreme High Purchase Amount (Row TXN-1014)**:
   - Value: `$2,450.00` (typical mean is ~$85.00).
   - Expected Flag: Both IQR and Z-score outlier.

2. **Extreme High Items Count (Row TXN-1029)**:
   - Value: `185` items (typical range 1–12).
   - Expected Flag: Both IQR and Z-score outlier.

3. **Data Quality / Typo in Customer Age (Row TXN-1042)**:
   - Value: `195` years.
   - Expected Flag: Impossible biological age; both IQR and Z-score flagged.

4. **Negative Value Typo in Purchase Amount (Row TXN-1055)**:
   - Value: `-$65.00`.
   - Expected Flag: Data-quality violation / negative outlier.

5. **Extreme High Session Duration (Row TXN-1068)**:
   - Value: `740.0` minutes (over 12 hours).
   - Expected Flag: Severe high outlier.

6. **Out-of-Range Rating Typo (Row TXN-1073)**:
   - Value: `9.5` on a 1.0–5.0 scale.
   - Expected Flag: Data validation error and high outlier.

7. **Missing Values (`NaN`)**:
   - `satisfaction_score` at Row TXN-1018 is missing (`NaN`).
   - `session_duration_min` at Row TXN-1049 is missing (`NaN`).

8. **Exact Duplicate Record**:
   - Row `TXN-1022` is duplicated verbatim once later in the table (Row `TXN-1022-DUP` or identical row content).

---

### Instructions for Uploading Real Public Datasets

Users can evaluate any public CSV dataset with numerical columns. Recommended sources:
- **Kaggle Datasets**: https://www.kaggle.com/datasets (e.g., California Housing, Wine Quality, Boston Housing, Iris)
- **UCI Machine Learning Repository**: https://archive.ics.uci.edu/
- **Data.gov**: https://data.gov/
- **World Bank Open Data**: https://data.worldbank.org/

When analyzing a real dataset:
1. Ensure the file is encoded in UTF-8 or standard Latin-1/CP1252.
2. Verify that header names are present on the first row.
3. Check the dataset's public license and respect terms of use.
