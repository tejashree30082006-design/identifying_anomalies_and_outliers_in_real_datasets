"""
Hypothesis Testing page (Module H).
Performs formal statistical hypothesis tests: Shapiro-Wilk test for normality
and Grubbs' test for a single outlier, emphasizing statistical assumptions and neutral inference.
"""

import streamlit as st
import pandas as pd

from core.hypothesis_testing import test_normality, test_grubbs_outlier
from utils.formatting import format_p_value
from utils.validation import get_numeric_columns


def render_hypothesis_page():
    """Render the Hypothesis Testing page."""
    st.header("🧪 Formal Statistical Hypothesis Testing")
    st.markdown(
        "Hypothesis testing provides inferential criteria to evaluate distributional conformity. "
        "A low p-value indicates statistical incompatibility with the null hypothesis, **never** proof of fraud or fabricated data."
    )

    from ui import check_dataset_and_analysis_ready
    if not check_dataset_and_analysis_ready():
        return

    working_df = st.session_state.get("working_df")

    num_cols = get_numeric_columns(working_df)
    if not num_cols:
        st.error("No numerical columns found in the active dataset.")
        return

    selected_cols = st.session_state.get("selected_columns", [])
    if not selected_cols:
        selected_cols = num_cols

    target_var = st.selectbox(
        "Select Variable for Hypothesis Testing",
        options=selected_cols,
        key="hyp_var_select"
    )

    if not target_var:
        return

    series = working_df[target_var]
    alpha = float(st.session_state.get("alpha", 0.05))

    tab_shapiro, tab_grubbs = st.tabs(["📊 1. Shapiro-Wilk Normality Test", "🎯 2. Grubbs' Single-Outlier Test"])

    # --- TAB 1: Shapiro-Wilk ---
    with tab_shapiro:
        st.subheader("Shapiro-Wilk Test for Distributional Normality")
        st.markdown(
            f"""
            - **Null Hypothesis ($H_0$):** The observations are drawn from a normally distributed population.
            - **Alternative Hypothesis ($H_1$):** The observations are not drawn from a normally distributed population.
            - **Significance Level ($\\alpha$):** `{alpha}`
            """
        )

        norm_res = test_normality(series, alpha=alpha)

        if not norm_res["is_valid"]:
            st.warning(f"⚠️ {norm_res['interpretation']}")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Test Statistic (W)", f"{norm_res['statistic']:.4f}")
            with c2:
                st.metric("P-Value", format_p_value(norm_res['p_value']))
            with c3:
                status_color = "🔴" if norm_res["reject_null"] else "🟢"
                st.metric("Decision", f"{status_color} {norm_res['decision'].split('(')[0].strip()}")

            if norm_res["reject_null"]:
                st.error(f"**Interpretation:** {norm_res['interpretation']}")
            else:
                st.success(f"**Interpretation:** {norm_res['interpretation']}")

            if norm_res.get("methodological_caveats"):
                st.info(f"ℹ️ {norm_res['methodological_caveats']}")

            with st.expander("📚 Statistical Guidelines & Limitations of Normality Tests"):
                st.markdown(
                    """
                    1. **Failing to reject $H_0$ is NOT proof of normality**: It merely indicates the sample size is insufficient to detect non-Gaussian features.
                    2. **High sensitivity in large samples**: With $N > 1,000$, virtually every real-world dataset rejects normality because no natural process is perfectly Gaussian.
                    3. **Impact on outlier detection**: Rejection of normality cautions against blind reliance on parametric 3σ Z-scores.
                    """
                )

    # --- TAB 2: Grubbs' Test ---
    with tab_grubbs:
        st.subheader("Grubbs' Test for a Single Suspected Outlier")
        st.markdown(
            f"""
            - **Null Hypothesis ($H_0$):** There are no outliers in the normally distributed sample.
            - **Alternative Hypothesis ($H_1$):** Exactly one observation is an outlier (at either minimum or maximum).
            - **Significance Level ($\\alpha$):** `{alpha}`
            """
        )

        grubbs_res = test_grubbs_outlier(series, alpha=alpha)

        if not grubbs_res["is_valid"]:
            st.warning(f"⚠️ {grubbs_res['interpretation']}")
        else:
            g1, g2, g3 = st.columns(3)
            with g1:
                st.metric("Grubbs' Statistic (G)", f"{grubbs_res['statistic_g']:.4f}")
            with g2:
                st.metric("Critical Value (G_crit)", f"{grubbs_res['critical_value']:.4f}")
            with g3:
                g_color = "🔴" if grubbs_res["reject_null"] else "🟢"
                st.metric("Decision", f"{g_color} {grubbs_res['decision'].split('(')[0].strip()}")

            st.markdown(
                f"- **Suspected Value Tested:** `{grubbs_res['suspect_value']:.2f}` ({grubbs_res['suspect_direction']})  \n"
                f"- **Approximate P-Value:** `{format_p_value(grubbs_res['p_value_approx'])}`"
            )

            if grubbs_res["reject_null"]:
                st.error(f"**Conclusion:** {grubbs_res['interpretation']}")
            else:
                st.success(f"**Conclusion:** {grubbs_res['interpretation']}")

            with st.expander("⚠️ Critical Assumptions & Masking Warning"):
                st.markdown(
                    """
                    - **Strict Normality Assumption:** Grubbs' test formally requires that non-outlier data follows an exact normal distribution. If the population is naturally skewed, Grubbs' test produces false positives.
                    - **Single Outlier Limitation (Masking Effect):** Grubbs' test evaluates at most one extreme point. If multiple outliers are present, their mutual inflation of the sample variance can prevent any single point from reaching statistical significance.
                    """
                )
