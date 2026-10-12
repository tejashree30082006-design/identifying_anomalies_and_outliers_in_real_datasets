# Can Statistics Detect Fake Data?

## Identifying Anomalies and Outliers in Real Datasets

**Academic Statistics Project** --- An interactive web application built
with Python and Streamlit to explore statistical anomalies and outliers
in real datasets.

> **Important:** Statistical methods can identify unusual observations,
> but an outlier alone does not prove that data was fabricated.

------------------------------------------------------------------------

## Quick Start

### 1. Open the project folder

``` powershell
cd statistics_fake_data_detector
```

### 2. Install the dependencies

``` powershell
pip install -r requirements.txt
```

### 3. Launch the application

``` powershell
python -m streamlit run app.py
```

Then open <http://localhost:8501> in your browser.

------------------------------------------------------------------------

## Project Overview

This project explores the question:

*Can standard statistical techniques help distinguish naturally
occurring variation from unusual or potentially anomalous observations?*

The application provides a workflow for uploading a CSV dataset,
examining its quality, calculating descriptive statistics, detecting
potential outliers, performing selected statistical tests, comparing
detection methods, and exporting results.

The results are intended for academic demonstrations and should be
interpreted in context.

------------------------------------------------------------------------

## Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  **CSV Upload**                      Upload a CSV dataset for analysis
                                      and validation.

  **Data Quality Audit**              Examine missing values, duplicate
                                      rows, infinite values, and constant
                                      columns where supported by the
                                      implementation.

  **Descriptive Statistics**          Calculate summary measures such as
                                      mean, median, mode, standard
                                      deviation, variance, IQR, skewness,
                                      kurtosis, and quartiles.

  **Distribution Analysis**           Explore distributions using
                                      visualizations such as histograms,
                                      KDE curves, and Q-Q plots where
                                      supported.

  **Outlier Detection**               Identify potential outliers using
                                      the Tukey IQR fence and Z-score
                                      methods.

  **Hypothesis Testing**              Apply supported statistical tests,
                                      including normality testing and
                                      other configured tests.

  **Results and Comparison**          Compare findings from different
                                      detection methods.

  **Export and Reporting**            Export analysis results and flagged
                                      records in supported formats.
  -----------------------------------------------------------------------

*Note: The available features depend on the current implementation. Keep
this table synchronized with the application code.*

------------------------------------------------------------------------

## Application Workflow

1.  Upload a CSV dataset.
2.  Preview the data and select numerical columns.
3.  Run the analysis.
4.  Review data quality and descriptive statistics.
5.  Explore distributions and potential outliers.
6.  Review statistical test results and compare detection methods.
7.  Export the results, where available.

------------------------------------------------------------------------

## Statistical Methods

### Outlier Detection

**Tukey IQR Fence**

The interquartile range is:

\[ IQR = Q_3 - Q_1 \]

Using the default multiplier (k = 1.5), the fences are:

-   Lower fence: (Q_1 - k `\cdot `{=tex}IQR)
-   Upper fence: (Q_3 + k `\cdot `{=tex}IQR)

Values outside these fences are flagged as potential outliers.

**Z-Score Method**

The Z-score indicates how far an observation is from the mean in
standard-deviation units:

\[ Z = `\frac{x - \mu}{\sigma}`{=tex} \]

A common threshold is (\|Z\| \> 3). The appropriate threshold depends on
the dataset and analysis assumptions. Z-scores can be misleading for
highly skewed data or data containing extreme values.

### Hypothesis Testing and Distribution Assessment

Depending on the implemented analysis, the project may include:

-   **Shapiro--Wilk test:** assesses evidence against the assumption
    that data follows a normal distribution.
-   **Grubbs' test:** tests for a single outlier under specific
    distributional assumptions.
-   **Independent-samples t-test:** compares the means of two groups
    when its assumptions are reasonably satisfied.
-   **Kernel Density Estimation (KDE):** provides a smoothed view of a
    distribution.
-   **Q-Q plot:** compares sample quantiles with theoretical
    normal-distribution quantiles.
-   **Skewness and excess kurtosis:** describe aspects of distribution
    shape.

Statistical tests have assumptions and limitations. Interpret p-values
and flagged observations alongside sample size, data quality, and
subject-matter knowledge.

------------------------------------------------------------------------

## Configuration

If exposed by the application, the sidebar may allow these parameters to
be adjusted:

  ------------------------------------------------------------------------
  Parameter                           Common default Purpose
  --------------------- ---------------------------- ---------------------
  IQR multiplier ((k))                         `1.5` Controls the
                                                     sensitivity of Tukey
                                                     fences.

  Z-score threshold                            `3.0` Controls how many
                                                     standard deviations
                                                     from the mean trigger
                                                     a flag.

  Significance level                          `0.05` Sets the significance
  ((`\alpha`{=tex}))                                 level for hypothesis
                                                     tests.
  ------------------------------------------------------------------------

Check the application controls and configuration files for the values
actually used by this version.

------------------------------------------------------------------------

## Project Structure

The following is the intended structure. Update it if your actual files
differ.

``` text
statistics_fake_data_detector/
├── app.py
├── requirements.txt
├── pytest.ini
├── config/
│   └── settings.py
├── core/
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── descriptive_stats.py
│   ├── distributions.py
│   ├── outlier_detection.py
│   ├── hypothesis_testing.py
│   ├── interpretation.py
│   ├── result_comparison.py
│   └── report_generator.py
├── ui/
│   ├── sidebar.py
│   ├── dataset_page.py
│   ├── data_quality_page.py
│   ├── descriptive_stats_page.py
│   ├── distribution_page.py
│   ├── outlier_page.py
│   ├── hypothesis_page.py
│   ├── results_page.py
│   └── export_page.py
├── data/
│   ├── demo_dataset.csv
│   └── README.md
├── reports/
└── tests/
```

------------------------------------------------------------------------

## Running Tests

If the project includes a pytest test suite, run:

``` powershell
python -m pytest
```

Review the test output to confirm which tests passed or failed.

------------------------------------------------------------------------

## Tech Stack

The project is built around the following Python ecosystem, depending on
the packages listed in `requirements.txt`:

-   **Python** --- application logic
-   **Streamlit** --- interactive web application
-   **Pandas** --- tabular data processing
-   **NumPy** --- numerical operations
-   **SciPy** --- statistical calculations and tests
-   **Matplotlib / Seaborn** --- data visualization

------------------------------------------------------------------------

## Limitations and Responsible Interpretation

-   A statistical outlier is an observation that differs from the
    general pattern; it is **not proof of fabricated data**.
-   Genuine observations may be flagged because of natural variation,
    rare events, measurement errors, or a mismatch between the data and
    the model assumptions.
-   Fabricated values may fall within expected ranges and remain
    undetected.
-   Results depend on data quality, sample size, distributional
    assumptions, and the selected detection thresholds.
-   Always investigate flagged observations and use domain knowledge
    before drawing conclusions.

------------------------------------------------------------------------

## Academic Use

This project is intended for educational exploration of data quality,
descriptive statistics, anomaly detection, and statistical inference. It
should not be used as the sole basis for accusing a person or
organization of fabricating data.
