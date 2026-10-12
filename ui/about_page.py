"""
About and Methodological Limitations page.
Provides academic background, mathematical definitions, epistemological analysis of fake data detection,
comparison tables, workflow explanations, and formal software references.
"""

import streamlit as st
from config.settings import (
    APP_TITLE,
    APP_SUBTITLE,
    VERSION,
    STATISTICAL_DISCLAIMER,
    METHOD_DESCRIPTIONS
)


def render_about_page():
    """Render the About, Methodological Foundations, and Limitations page."""
    st.header("ℹ️ About, Methodology & Statistical Foundations")
    st.subheader(f"{APP_TITLE} (v{VERSION})")
    st.caption(APP_SUBTITLE)

    st.warning(f"⚠️ **Core Scientific Axiom:** {STATISTICAL_DISCLAIMER}")

    tab_question, tab_math, tab_workflow, tab_limits, tab_refs = st.tabs([
        "🔬 Research Question & Philosophy",
        "📐 Mathematical Foundations",
        "🔄 Analytical Workflow",
        "⚠️ Assumptions & Limitations",
        "📚 Academic References"
    ])

    with tab_question:
        st.markdown("### Can Statistics Detect Fake Data?")
        st.markdown(
            """
            The central thesis of this project asks: **To what degree can classical and exploratory statistical methods 
            identify fabricated or fraudulent records?**

            #### The Statistical Boundary
            Statistical outlier detection operates exclusively on **distributional distance**:
            - It measures how improbable an observation is under an assumed probability model (e.g., standard normal curve).
            - It calculates how far a value lies beyond empirical quartiles (Tukey's IQR fences).

            #### Why Outlier Detection Alone Cannot Prove Fraud
            1. **Legitimate Black Swans:** Real-world systems generate extreme values through power-law dynamics, rare market shocks, or natural events. Labeling these as 'fake' commits a fundamental error of induction.
            2. **Smart Fabricators:** A sophisticated actor fabricating data will intentionally inject values near the mean or median to avoid extreme fences, rendering standard outlier detectors blind to the manipulation.
            3. **Data Entry Typos vs. Fraud:** An age of `195` is clearly invalid, but it is almost certainly a clerical keystroke typo rather than an intentional fraud conspiracy.
            
            Therefore, statistical detection serves as a **screening filter** to prioritize audit candidates, never as self-sufficient proof of intentional deception.
            """
        )

        st.divider()

        st.markdown("### ⚖️ Outliers vs. Fabricated Data")
        st.markdown(
            """
            | Concept | Statistical Reality | Audit Implication |
            | :--- | :--- | :--- |
            | **Statistical Outlier** | Extreme distance from distributional center | Valid realization of heavy-tailed distribution |
            | **Measurement Error** | Physical impossibility or instrument noise | Correction or exclusion justified |
            | **Suspicious Observation** | Nonconformity across multiple models | Prioritized candidate for deeper audit |
            | **Confirmed Fabrication** | Unverifiable by outlier statistics alone | Requires forensic audit trails & origin verification |
            """
        )

        st.divider()

        st.markdown("### 🎯 Why Unusual Observations Matter")
        st.markdown(
            """
            In empirical sciences, financial systems, and public datasets, unusual observations can point to:
            - **Legitimate extreme natural phenomena** (e.g., rare market rallies or severe weather events).
            - **Measurement, recording, or logging artifacts** (e.g., sensor malfunction, comma-versus-dot decimal typos).
            - **Data entry errors** (e.g., recording age as 195 instead of 19.5).
            - **Intentional manipulation or fabrication** (e.g., simulated survey responses, forged account books).
            """
        )

    with tab_math:
        st.markdown("### Mathematical Formulation of Implemented Methods")
        for method, desc in METHOD_DESCRIPTIONS.items():
            st.markdown(f"#### {method.upper()}")
            st.markdown(desc)
            st.divider()

    with tab_workflow:
        st.markdown("### 🔄 User Workflow: From Upload to Audit")
        st.info(
            "**Step 1: Upload CSV** → **Step 2: Preview Dataset** → **Step 3: Select Numerical Columns** → "
            "**Step 4: Click 'Analyze Dataset'** → **Step 5: View Statistics & Charts** → **Step 6: Investigate Outliers** → **Step 7: Download Results**"
        )
        st.markdown(
            """
            1. **Data Hygiene:** Audit missing cells, duplicate records, infinite values, and data types.
            2. **Descriptive Stats:** Quantify dispersion ($s, ddof=1$), skewness, kurtosis, and quartiles.
            3. **Outlier Fences:** Execute Tukey IQR fences and sample Z-scores.
            4. **Hypothesis Tests:** Evaluate Shapiro-Wilk normality and Grubbs' single outlier test.
            5. **Export Dossier:** Synthesize method agreement into four mutually exclusive categories and export full audit reports.
            """
        )

    with tab_limits:
        st.markdown("### Critical Assumptions & Boundary Conditions")
        st.markdown(
            """
            - **Normality Dependence:** Z-scores and Grubbs' test assume Gaussian distributions. When applied to skewed, bimodal, or heavy-tailed phenomena, they generate severe rates of false alarms.
            - **Masking & Swamping:** 
              - *Masking:* Multiple severe outliers inflate the sample standard deviation $s$, artificially suppressing Z-scores for other anomalies.
              - *Swamping:* An extreme outlier shifts the sample mean $\\bar{x}$, causing innocent central values to appear unusually distant.
            - **Sample Size Restrictions:** Shapiro-Wilk requires $3 \\le N \\le 5,000$. Very large sample sizes will reject normality even for practically indistinguishable departures.
            - **Zero-Variance Degeneracy:** Constant columns collapse fences and render division by zero in variance formulas.
            """
        )

    with tab_refs:
        st.markdown("### Academic References & Statistical Foundations")
        st.markdown(
            """
            1. **Tukey, J. W. (1977).** *Exploratory Data Analysis*. Addison-Wesley. (Foundational specification of boxplots and 1.5×IQR fences).
            2. **Grubbs, F. E. (1950).** *Sample criteria for testing outlying observations*. Annals of Mathematical Statistics, 21(1), 27–58.
            3. **Shapiro, S. S., & Wilk, M. B. (1965).** *An analysis of variance test for normality (complete samples)*. Biometrika, 52(3/4), 591–611.
            4. **Barnett, V., & Lewis, T. (1994).** *Outliers in Statistical Data* (3rd ed.). John Wiley & Sons.
            5. **Hyndman, R. J., & Fan, Y. (1996).** *Sample quantiles in statistical packages*. The American Statistician, 50(4), 361–365.
            """
        )
