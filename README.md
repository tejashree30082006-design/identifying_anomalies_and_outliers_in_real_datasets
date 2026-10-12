# Can Statistics Detect Fake Data?
### Identifying Anomalies and Outliers in Real Datasets Using Statistical Methodology

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-pytest%20(31%20passed)-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT%20Academic-green.svg)]()

---

## 📌 Executive Summary & Academic Framing

This project develops an interactive, statistically rigorous application to investigate the research question:  
**To what extent can exploratory and classical statistical methods distinguish between ordinary variation, data-quality artifacts, statistical outliers, and potentially suspicious observations?**

Designed for college statistics demonstration and faculty evaluation, the application provides an end-to-end analytical pipeline: data ingestion, data quality audits, descriptive metrics, probability distribution assessments, Tukey Interquartile Range (IQR) fences, parametric Z-score detection, formal hypothesis testing (Shapiro-Wilk and Grubbs' test), method concordance analysis without double-counting, source-row provenance tracking, and reproducible report generation.

> ### ⚠️ Critical Methodological Disclaimer
> **Statistical outlier detection identifies observations that deviate from an assumed distribution or the bulk of the sample. It cannot and does not prove intentional data fabrication or fraud.**  
> An outlier may reflect a legitimate rare event (e.g., high-volume transaction), instrument noise, a recording typo (e.g., biological age of 195), heavy-tailed distribution dynamics, or model misspecification. Statistical flags prioritize candidates for audit; confirming fraud requires independent forensic verification, audit logs, and domain context.

---

## 📑 Table of Contents
1. [Core Features](#-core-features)
2. [Statistical Distinctions: Outliers vs. Fabricated Data](#-statistical-distinctions-outliers-vs-fabricated-data)
3. [Exact Project Structure](#-exact-project-structure)
4. [Mathematical Formulations & Assumptions](#-mathematical-formulations--assumptions)
5. [Installation & Setup](#-installation--setup)
6. [Running the Application](#-running-the-application)
7. [Running Automated Tests](#-running-automated-tests)
8. [End-to-End Demonstration Walkthrough](#-end-to-end-demonstration-walkthrough)
9. [Analyzing Real Public Datasets](#-analyzing-real-public-datasets)
10. [Assumptions, Limitations & Edge Cases](#-assumptions-limitations--edge-cases)

---

## 🌟 Core Features

- **Multi-Encoding CSV Ingestion:** Automatic fallback across UTF-8, UTF-8-sig, Latin-1, CP1252, and ISO-8859-1 with file-size verification (50 MB limit).
- **Data Quality & Hygiene Audit:** Audits missing values, duplicate records, infinite values, constant columns, and column typing without altering source data.
- **Descriptive Statistics:** Calculates valid/missing counts, mean, median, mode, min, max, range, sample variance ($s^2, ddof=1$), sample standard deviation ($s, ddof=1$), Q1, Q3, IQR, sample skewness, and Fisher's excess kurtosis.
- **Distribution Analysis:** Histograms with estimated normal PDF overlays, Tukey boxplots, and standard normal Q-Q probability plots.
- **Tukey IQR Rule ($k=1.5$):** Resistant non-parametric outlier detection; handles zero-IQR edge cases.
- **Standard Z-Score ($|Z| > 3.0\sigma$):** Parametric dispersion detection using unbiased sample standard deviation.
- **Mutually Exclusive Comparison:** Synthesizes detections into four non-overlapping categories (Both, IQR Only, Z-Score Only, Neither), strictly preventing double-counting.
- **Formal Hypothesis Testing:**
  - *Shapiro-Wilk Normality Test* ($H_0$: normality vs. $H_1$: non-normality) with sample-size boundary checks ($3 \le N \le 5,000$).
  - *Grubbs' Single-Outlier Test* ($H_0$: no outlier vs. $H_1$: exactly one outlier) using Student's $t$ critical values.
- **Full Source Row Traceability:** Stable internal identifiers (`_record_id`, `_original_row_number`) allow users to inspect the entire source record behind any flagged value.
- **Export Deliverables:** Downloadable CSV tables (flagged records, statistics, full detection), sanitized JSON summaries, and Markdown audit reports.

---

## ⚖️ Statistical Distinctions: Outliers vs. Fabricated Data

The application formally distinguishes among four empirical states:

| Category | Statistical Definition | Typical Origin | Recommended Action |
| :--- | :--- | :--- | :--- |
| **1. Statistically Unusual** | Extreme observation in tail of empirical distribution | Heavy-tailed population, high natural variance | Retain in sample; use resistant estimators |
| **2. Data-Quality Error** | Value outside physically possible bounds (e.g., Age = 195) | Typo, transmission loss, comma/dot delimiter error | Document error; exclude or correct via source logs |
| **3. Potentially Suspicious** | Multi-method divergence (flagged by Both IQR and Z-Score) | Severe anomaly inconsistent with sample core | Flag for prioritized domain audit |
| **4. Fabricated / Fake Data** | Deliberately manufactured or forged records | Fraudulent manipulation | **Cannot be confirmed by outlier statistics alone** |

---

## 📁 Exact Project Structure

```
statistics_fake_data_detector/
│
├── app.py                     # Streamlit application entry point & router
├── requirements.txt           # Core Python package dependencies
├── README.md                  # Comprehensive academic documentation
├── .gitignore                 # Standard Python/IDE ignore rules
├── pytest.ini                 # Pytest configuration settings
│
├── config/
│   └── settings.py            # Global constants, thresholds, paths & disclaimers
│
├── core/
│   ├── __init__.py
│   ├── data_loader.py         # Multi-encoding CSV loading & provenance assignment
│   ├── data_quality.py        # Missing values, duplicates & non-finite audits
│   ├── descriptive_stats.py   # Unbiased summary metrics, quartiles & dispersion
│   ├── distributions.py       # Empirical symmetry, tails & distribution summaries
│   ├── outlier_detection.py   # Tukey IQR fences & sample Z-score detection
│   ├── hypothesis_testing.py  # Shapiro-Wilk & Grubbs' test implementations
│   ├── result_comparison.py   # 4 mutually exclusive categories & deduplication
│   ├── interpretation.py      # Rule-based objective statistical interpretations
│   └── report_generator.py    # Native Markdown audit report compiler
│
├── ui/
│   ├── __init__.py
│   ├── sidebar.py             # Navigation & parameter control widgets
│   ├── home.py                # Project overview & academic disclaimers
│   ├── dataset_page.py        # CSV upload & demo dataset loading
│   ├── data_quality_page.py   # Data quality audit & hygiene dashboard
│   ├── descriptive_stats_page.py # Descriptive metrics tables & narratives
│   ├── distribution_page.py   # Histograms, fitted normal PDFs, boxplots, Q-Q plots
│   ├── outlier_page.py        # Independent IQR & Z-Score diagnostic pages
│   ├── hypothesis_page.py     # Formal normality & Grubbs tests
│   ├── results_page.py        # Combined method comparison & row inspection
│   ├── export_page.py         # CSV, JSON, and Markdown report downloads
│   └── about_page.py          # Mathematical foundations & limitations
│
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Input sanity checking & sample-size constraints
│   ├── plotting.py            # Polished Matplotlib & Seaborn diagnostic figures
│   ├── formatting.py          # Number, p-value, and percentage formatters
│   └── export_utils.py        # CSV converters & JSON-safe serializers
│
├── data/
│   ├── README.md              # Dataset variable dictionary & ground truth log
│   └── demo_dataset.csv       # Synthetic retail/audit dataset (71 rows)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py    # Loading, encoding, empty/malformed handling tests
│   ├── test_data_quality.py   # Missingness, duplicates, infinite checks tests
│   ├── test_descriptive_stats.py # Summary metrics, quartiles, constant column tests
│   ├── test_outlier_detection.py # IQR fences, Z-scores, bipolar extremes tests
│   ├── test_hypothesis_testing.py # Shapiro-Wilk & Grubbs sample size tests
│   ├── test_result_comparison.py # 4-category mutually exclusive tests
│   └── test_exports.py        # CSV, JSON-safe, and Markdown report export tests
│
└── reports/
    └── README.md              # Report storage directory documentation
```

---

## 📐 Mathematical Formulations & Assumptions

### 1. Tukey's Interquartile Range (IQR) Rule
- **First Quartile ($Q_1$):** 25th percentile (linear interpolation, Type 7).
- **Third Quartile ($Q_3$):** 75th percentile.
- **Interquartile Range:**
  $$\text{IQR} = Q_3 - Q_1$$
- **Inner Fences:**
  $$\text{Lower Fence} = Q_1 - k \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + k \times \text{IQR} \quad (k=1.5)$$
- *Properties:* Non-parametric and resistant to extreme values. If $\text{IQR} = 0$, fences collapse to a single point; the detector explains this limitation rather than treating every variation as an outlier.

### 2. Standard Sample Z-Score Method
- **Sample Mean ($\bar{x}$) & Sample Standard Deviation ($s, ddof=1$):**
  $$\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i, \quad s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2}$$
- **Standardized Score ($Z_i$):**
  $$Z_i = \frac{x_i - \bar{x}}{s}$$
- **Outlier Criterion:** Flag observation $i$ if $|Z_i| > \text{threshold}$ (default $3.0\sigma$).
- *Properties:* Parametric metric assuming approximate normality. Subject to **masking** (multiple extreme values inflate $s$, lowering $Z$-scores) and **swamping** (outliers pull $\bar{x}$, flagging innocent central values).

### 3. Shapiro-Wilk Normality Test
- Tests $H_0: \mathbf{x} \sim \mathcal{N}(\mu, \sigma^2)$ against $H_1: \mathbf{x} \not\sim \mathcal{N}(\mu, \sigma^2)$.
- Valid for sample sizes $3 \le N \le 5,000$.
- *Critical Caveat:* Failing to reject $H_0$ ($p \ge \alpha$) **does not prove** the population is normal. For very large $N$, the test has extreme statistical power and will reject $H_0$ for trivial, practically negligible non-normalities.

### 4. Grubbs' Test for Outliers (Maximum Normed Residual)
- Tests $H_0$: No outliers exist in the normally distributed sample vs. $H_1$: Exactly one outlier exists (at minimum or maximum).
- **Grubbs' Statistic ($G$):**
  $$G = \frac{\max_i |x_i - \bar{x}|}{s}$$
- **Two-Sided Critical Threshold ($G_{\text{crit}}$) at Significance $\alpha$:**
  $$G_{\text{crit}} = \frac{N-1}{\sqrt{N}} \sqrt{\frac{t_{\alpha/(2N),\, N-2}^2}{N - 2 + t_{\alpha/(2N),\, N-2}^2}}$$
  where $t_{\alpha/(2N),\, N-2}$ is the upper critical value of the Student's $t$-distribution with $N-2$ degrees of freedom.

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed locally.
- Web browser (Chrome, Edge, Firefox, or Safari).

### Windows Setup (PowerShell)
```powershell
# 1. Navigate to the project root directory
cd statistics_fake_data_detector

# 2. Create a virtual environment (optional if global Python packages are installed)
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

### macOS / Linux Setup (Bash / Zsh)
```bash
# 1. Navigate to the project root directory
cd statistics_fake_data_detector

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will start locally and display the URL:
```
Local URL: http://localhost:8501
```
Open this address in any modern web browser. The application is completely self-contained and operates **100% offline without internet access**.

---

## 🧪 Running Automated Tests

Run the complete test suite using `pytest`:

```bash
python -m pytest
```

### Test Coverage Summary (31 Tests, 100% Passing)
The test suite validates all 20 required edge cases and analytical workflows:
1. `test_clean_data_no_outliers`: Normally clustered data has zero flagged points.
2. `test_large_positive_outlier`: Detects extreme positive values.
3. `test_small_negative_outlier`: Detects extreme negative values.
4. `test_both_positive_and_negative_extremes`: Bipolar outlier detection.
5. `test_iqr_and_zscore_disagreement`: Tests divergence between resistant IQR and parametric Z-score.
6. `test_constant_column`: Handles zero-variance constant columns without division by zero.
7. `test_missing_value_summary`: Missing value counts and complete-case calculations.
8. `test_infinite_values_and_constant_columns`: Handles $+\infty$ and $-\infty$ safely.
9. `test_all_missing_column`: All-NaN columns handled gracefully without crashes.
10. `test_non_numeric_and_empty_dataset`: Separates text/categorical from numeric variables.
11. `test_prepare_dataset_with_duplicates`: Stable unique provenance for duplicate rows.
12. `test_load_empty_csv`: Empty CSV files raise informative errors.
13. `test_load_malformed_csv`: Malformed CSV lines caught cleanly.
14. `test_small_sample_size`: Handles $N=1$ and $N=2$ safely without runtime crashes.
15. `test_shapiro_sample_sizes` & `test_grubbs_constant_and_small_sample`: Sample size bounds enforced.
16. `test_normally_distributed_sample` & `test_strongly_non_normal_sample`: Validates hypothesis test power.
17. `test_mutually_exclusive_categories_no_double_counting`: Exact category summation preventing double counting.
18. `test_csv_exports`, `test_sanitize_for_json`, `test_markdown_report_generation`: Validates all export formats.
19. `test_prepare_dataset_preserves_original_and_adds_provenance`: Verifies uploaded DataFrame is never modified.
20. `test_flagged_records_map_to_correct_row_ids`: Traces flagged points back to source rows.
21. `test_first_user_csv_housing_dataset`: Full dynamic pipeline on Housing CSV (40 rows, 5 columns, specific price/lot outliers).
22. `test_second_user_csv_sensor_dataset`: Full dynamic pipeline on Sensor CSV (60 rows, 5 columns, specific temperature/pressure outliers).
23. `test_dynamic_variation_between_datasets`: Proves statistics, quartiles, and flagged row indices strictly adapt between different uploaded files.

---

## 📖 Primary User Workflow & Demonstration Walkthrough

The primary workflow requires the user to upload their own CSV file:

$$\text{\bf Upload CSV} \longrightarrow \text{\bf Preview Dataset} \longrightarrow \text{\bf Select Numerical Columns} \longrightarrow \text{\bf Click Analyze Dataset} \longrightarrow \text{\bf View Statistics \& Charts} \longrightarrow \text{\bf Investigate Outliers} \longrightarrow \text{\bf Download Results}$$

Follow this step-by-step procedure:

1. **Launch the Application:** Run `streamlit run app.py` and open `http://localhost:8501`.
2. **Review Home Page (Module A):** Notice the prominent uploader: *"Upload your CSV dataset to begin statistical analysis."*
3. **Upload Your Custom CSV (Module B):**
   - Drag and drop your `.csv` file onto the uploader.
   - The application validates file size, encoding (UTF-8, Latin-1, CP1252), and schema.
   - Dynamic preview displays:
     - Uploaded filename, row count, and column count.
     - First 10 rows with provenance row numbers.
     - Column data types, missing-value counts, and duplicate-row summary.
     - Available numerical columns.
4. **Select Numerical Columns & Click "Analyze Dataset":**
   - Select one or multiple numerical columns using the multiselect widget.
   - Click the prominent **"🚀 Analyze Dataset"** button.
   - The application dynamically computes descriptive statistics, dispersion metrics, IQR fences, Z-scores, distributions, and hypothesis tests.
5. **Inspect Data Quality (Module C):**
   - Navigate to **"3. Data Quality Audit"** to review missingness charts and duplicate records.
6. **Analyze Descriptive Statistics (Module D):**
   - Navigate to **"4. Descriptive Statistics"** to review the dynamically generated summary table ($s, ddof=1$, Fisher's excess kurtosis, skewness, quartiles) and plain-English narratives.
7. **Inspect Empirical Distributions (Module E):**
   - Navigate to **"5. Distribution Analysis"** to inspect histograms with fitted Gaussian PDF overlays, boxplots, and normal Q-Q probability plots.
8. **Perform Outlier Detection (Module F):**
   - Navigate to **"6. Outlier Detection (IQR & Z-Score)"** to view Tukey fences ($k=1.5$), signed Z-score scatter plots with $\pm 3.0\sigma$ boundary lines, and flagged observation tables.
9. **Conduct Formal Hypothesis Testing (Module G):**
   - Navigate to **"7. Hypothesis Testing"** to review Shapiro-Wilk normality decisions ($p$-values) and Grubbs' single-outlier test against Student's $t$ critical values.
10. **Synthesize Results & Investigate Rows (Module H & J):**
    - Navigate to **"8. Results & Comparison"** to review the 4 mutually exclusive categories chart (*Both*, *IQR Only*, *Z-Score Only*, *Neither*).
    - Use the **Deep-Dive Source Row Inspection** dropdown to view the entire original record behind any flagged point.
11. **Export Analytical Dossier (Module K & L):**
    - Navigate to **"9. Export & Reporting"** to download the Flagged Records CSV, Descriptive Statistics CSV, Full Detection Results CSV, sanitized JSON summary, and Markdown audit report (`.md`).
12. **Start a Fresh Analysis:**
    - Click **"🔄 Upload a Different CSV / Start Fresh Analysis"** at any time to upload a new dataset without restarting the server.

---

## 🌐 Analyzing Real Public Datasets

Users can import any real tabular CSV dataset. Recommended open repositories:
- **Kaggle Datasets:** [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) (e.g., California Housing, Wine Quality, Credit Card Fraud Detection).
- **UCI Machine Learning Repository:** [https://archive.ics.uci.edu/](https://archive.ics.uci.edu/) (e.g., Iris, Boston Housing, Auto MPG).
- **Data.gov:** [https://data.gov/](https://data.gov/) (Federal open datasets).

### Import Guidelines
1. Ensure the dataset is a standard CSV file with column headers on row 1.
2. File size must be under 50 MB.
3. The dataset should contain at least one continuous numerical variable.
4. The application never transmits data externally; all processing occurs strictly in local computer memory.

---

## ⚠️ Assumptions, Limitations & Edge Cases

1. **Scientific Neutrality:** Outlier detection identifies distributional rarity, not intentional fraud. Fabricators can generate fake numbers clustered near the sample mean to avoid detection; statistical outlier models are blind to such manipulations.
2. **Normality Assumption Sensitivity:** Parametric metrics (Z-scores, Grubbs' test) assume Gaussian data. Applying 3σ thresholds to power-law or log-normal distributions causes high false-positive rates.
3. **Masking & Swamping Effects:** Multiple severe outliers inflate the sample standard deviation $s$, artificially suppressing Z-scores for secondary anomalies (masking).
4. **Complete-Case Estimation Policy:** Missing values are isolated and excluded from numerical point calculations. Original source rows are preserved verbatim.
5. **Quantile Convention:** Quartiles ($Q_1, Q_3$) use linear interpolation (equivalent to R Type 7 and NumPy default).

---

## 📜 Academic Integrity & Software License

Developed as an open academic project for university statistics coursework and research demonstration.  
Released under the **MIT License**.
#   i d e n t i f y i n g _ a n o m a l i e s _ a n d _ o u t l i e r s _ i n _ r e a l _ d a t a s e t s  
 