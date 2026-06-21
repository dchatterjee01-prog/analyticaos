import streamlit as st
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.api as sms
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR,
    TEXT_COLOR, ACCENT_COLOR, SUCCESS_BG, SUCCESS_COLOR, WARNING_BG, WARNING_COLOR
)

def _inject_css():
    st.markdown(f"""
    <style>
      .metric-card {{ background: {SURFACE_COLOR}; border: 1px solid {PRIMARY_COLOR}33; border-radius: 10px; padding: 1rem; text-align: center; }}
      .metric-val {{ font-size: 1.6rem; font-weight: 800; color: {ACCENT_COLOR}; }}
      .metric-lbl {{ font-size: 0.72rem; color: {TEXT_COLOR}88; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }}
      .verdict-sig {{ background: {SUCCESS_BG}; color: {SUCCESS_COLOR}; border: 1px solid {SUCCESS_COLOR}; border-radius: 20px; padding: 0.25rem 0.9rem; font-size: 0.8rem; font-weight: 600; display: inline-block; }}
      .verdict-nosig {{ background: {WARNING_BG}; color: {WARNING_COLOR}; border: 1px solid {WARNING_COLOR}; border-radius: 20px; padding: 0.25rem 0.9rem; font-size: 0.8rem; font-weight: 600; display: inline-block; }}
    </style>
    """, unsafe_allow_html=True)

def _metric_card(col, value, label):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{value}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

def _verdict_badge(p_value, alpha=0.05):
    if pd.isna(p_value): return '<span class="verdict-nosig">⚪ Undetermined (NaN)</span>'
    if p_value < alpha: return f'<span class="verdict-sig">✅ Statistically Significant (p={p_value:.4f} &lt; {alpha})</span>'
    return f'<span class="verdict-nosig">⚪ Not Significant (p={p_value:.4f} &ge; {alpha})</span>'

def _standard_ab_test(df: pd.DataFrame, num_cols: list, cat_cols: list):
    st.markdown("### ⚙️ Configure Standard A/B Test")
    
    if not cat_cols:
        st.warning("You need at least one categorical column to define Test/Control groups.")
        return

    c1, c2 = st.columns(2)
    group_col = c1.selectbox("Group / Variant Column:", options=cat_cols, key="ab_group_col")
    metric_col = c2.selectbox("Metric to Measure:", options=num_cols, key="ab_metric_col")

    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 2:
        st.error(f"'{group_col}' has fewer than 2 unique variants. Cannot run A/B test.")
        return

    g1, g2 = st.columns(2)
    control = g1.selectbox("Control Group (A):", options=groups, index=0, key="ab_ctrl")
    treatment = g2.selectbox("Treatment Group (B):", options=groups, index=min(1, len(groups)-1), key="ab_treat")

    if control == treatment:
        st.warning("Control and Treatment must be different groups.")
        return

    test_type = st.radio("Metric Type:", ["Continuous (Means / T-Test)", "Binary Conversion (Proportions / Z-Test)"], horizontal=True)
    
    if st.button("🧪 Run A/B Test", type="primary"):
        data_c = df[df[group_col] == control][metric_col].dropna()
        data_t = df[df[group_col] == treatment][metric_col].dropna()

        if len(data_c) < 2 or len(data_t) < 2:
            st.error("Not enough data in one or both groups.")
            return

        st.divider()
        m1, m2, m3, m4 = st.columns(4)

        if "Continuous" in test_type:
            if data_c.nunique() <= 1 and data_t.nunique() <= 1:
                st.error("⚠️ Both groups have identical values (zero variance). Cannot compute statistical significance.")
                return

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stat, p_val = stats.ttest_ind(data_c, data_t, equal_var=False)
            
            mean_c, mean_t = data_c.mean(), data_t.mean()
            uplift = ((mean_t - mean_c) / mean_c) * 100 if mean_c != 0 else 0
            
            _metric_card(m1, f"{mean_c:.3f}", "Control Mean")
            _metric_card(m2, f"{mean_t:.3f}", "Treatment Mean")
            _metric_card(m3, f"{uplift:+.2f}%", "Relative Uplift")
            _metric_card(m4, f"{p_val:.4f}" if not pd.isna(p_val) else "NaN", "P-Value")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(_verdict_badge(p_val), unsafe_allow_html=True)
            
        else:
            succ_c, n_c = (data_c > 0).sum(), len(data_c)
            succ_t, n_t = (data_t > 0).sum(), len(data_t)
            
            if n_c == 0 or n_t == 0:
                st.error("Zero observations in group.")
                return
            
            if (succ_c == 0 or succ_c == n_c) and (succ_t == 0 or succ_t == n_t):
                st.error("⚠️ Both groups have exactly 0% or 100% conversion rates (zero variance). Cannot compute Z-Statistic.")
                return

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stat, p_val = proportions_ztest([succ_t, succ_c], [n_t, n_c], alternative='two-sided')
            
            rate_c = succ_c / n_c
            rate_t = succ_t / n_t
            uplift = ((rate_t - rate_c) / rate_c) * 100 if rate_c != 0 else 0

            _metric_card(m1, f"{rate_c:.2%}", "Control Conv. Rate")
            _metric_card(m2, f"{rate_t:.2%}", "Treatment Conv. Rate")
            _metric_card(m3, f"{uplift:+.2f}%", "Relative Uplift")
            _metric_card(m4, f"{p_val:.4f}" if not pd.isna(p_val) else "NaN", "P-Value")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(_verdict_badge(p_val), unsafe_allow_html=True)

def _power_analysis():
    st.markdown("### ⚡ Sample Size & Power Calculator")
    st.info("Determine how many users/rows you need in each group before running an experiment to detect a true effect.")

    calc_type = st.radio("Metric Type for Sizing:", ["Continuous (Means)", "Binary (Proportions)"], horizontal=True, key="pwr_type")

    c1, c2, c3, c4 = st.columns(4)
    alpha = c1.number_input("Significance Level (α):", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="pwr_alpha")
    power = c2.number_input("Statistical Power (1-β):", min_value=0.50, max_value=0.99, value=0.80, step=0.05, key="pwr_power")

    if "Continuous" in calc_type:
        baseline_mean = c3.number_input("Baseline Mean:", value=100.0, key="pwr_c_mean")
        baseline_std = c4.number_input("Baseline Std Dev:", min_value=0.1, value=15.0, key="pwr_c_std")
        mde = st.number_input("Minimum Detectable Effect (Absolute change):", min_value=0.1, value=5.0, key="pwr_c_mde")

        if st.button("⚡ Calculate Sample Size", type="primary", key="pwr_c_btn"):
            effect_size = mde / baseline_std
            n_required = sms.TTestIndPower().solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
            
            st.divider()
            n_ceil = int(np.ceil(n_required))
            st.success(f"**Required Sample Size:** ~{n_ceil:,} per group (Total: ~{n_ceil * 2:,})")
            st.caption(f"Assumes Cohen's d effect size of {effect_size:.3f}")

    else:
        baseline_rate = c3.number_input("Baseline Conv. Rate:", min_value=0.001, max_value=0.999, value=0.050, step=0.005, format="%.3f", key="pwr_b_rate")
        mde_rel = c4.number_input("MDE (Relative % uplift):", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, key="pwr_b_mde")
        
        if st.button("⚡ Calculate Sample Size", type="primary", key="pwr_b_btn"):
            rate2 = baseline_rate * (1 + (mde_rel / 100.0))
            if rate2 >= 1.0:
                st.error("Target conversion rate exceeds 100%. Please lower your baseline rate or MDE.")
            else:
                effect_size = sms.proportion_effectsize(baseline_rate, rate2)
                n_required = sms.NormalIndPower().solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
                
                st.divider()
                n_ceil = int(np.ceil(n_required))
                st.success(f"**Required Sample Size:** ~{n_ceil:,} per group (Total: ~{n_ceil * 2:,})")
                st.caption(f"Targeting a new conversion rate of {rate2:.2%} (Cohen's h effect size: {abs(effect_size):.3f})")

def _cuped_analysis(df: pd.DataFrame, num_cols: list, cat_cols: list):
    st.markdown("### 📉 CUPED Variance Reduction")
    st.info(
        "**CUPED** uses a pre-experiment metric (covariate) to explain away natural "
        "variance in your target metric. This shrinks the noise, making your A/B test "
        "more powerful without requiring more users."
    )

    if not cat_cols or len(num_cols) < 2:
        st.warning("Requires 1 categorical column (Groups) and at least 2 numeric columns (Metric + Covariate).")
        return

    c1, c2, c3 = st.columns(3)
    group_col = c1.selectbox("Variant Column:", options=cat_cols, key="cuped_grp")
    metric_col = c2.selectbox("Target Metric (During Test):", options=num_cols, key="cuped_met")
    
    covar_options = [c for c in num_cols if c != metric_col]
    covar_col = c3.selectbox("Pre-Experiment Covariate:", options=covar_options, key="cuped_cov")

    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 2:
        st.error("Not enough variants to compare.")
        return

    g1, g2 = st.columns(2)
    control = g1.selectbox("Control Group:", options=groups, index=0, key="cuped_ctrl")
    treatment = g2.selectbox("Treatment Group:", options=groups, index=min(1, len(groups)-1), key="cuped_treat")

    if st.button("📉 Run CUPED Analysis", type="primary", key="cuped_btn"):
        clean_df = df[[group_col, metric_col, covar_col]].dropna().copy()
        
        data_c = clean_df[clean_df[group_col] == control]
        data_t = clean_df[clean_df[group_col] == treatment]

        if len(data_c) < 2 or len(data_t) < 2:
            st.error("Not enough valid data after dropping missing values.")
            return

        if clean_df[metric_col].nunique() <= 1 or clean_df[covar_col].nunique() <= 1:
            st.error("⚠️ Zero variance detected in target or covariate. Cannot compute CUPED.")
            return

        cov_matrix = np.cov(clean_df[metric_col], clean_df[covar_col])
        covariance = cov_matrix[0, 1]
        var_covar = np.var(clean_df[covar_col], ddof=1)
        
        if var_covar == 0:
            st.error("Pre-experiment covariate has zero variance.")
            return
            
        theta = covariance / var_covar
        mean_covar = clean_df[covar_col].mean()

        clean_df["CUPED_Metric"] = clean_df[metric_col] - theta * (clean_df[covar_col] - mean_covar)

        cuped_c = clean_df[clean_df[group_col] == control]["CUPED_Metric"]
        cuped_t = clean_df[clean_df[group_col] == treatment]["CUPED_Metric"]
        orig_c = data_c[metric_col]
        orig_t = data_t[metric_col]

        orig_var = clean_df[metric_col].var()
        cuped_var = clean_df["CUPED_Metric"].var()
        var_reduction = ((orig_var - cuped_var) / orig_var) * 100

        st.divider()
        st.markdown("#### 1. Variance Reduction")
        m1, m2, m3 = st.columns(3)
        _metric_card(m1, f"{orig_var:,.2f}", "Original Variance")
        _metric_card(m2, f"{cuped_var:,.2f}", "CUPED Variance")
        _metric_card(m3, f"{var_reduction:.1f}%", "Variance Reduction")

        if var_reduction < 0:
            st.warning("Variance increased. Your covariate is likely not correlated with your target metric.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            orig_stat, orig_pval = stats.ttest_ind(orig_c, orig_t, equal_var=False)
            cuped_stat, cuped_pval = stats.ttest_ind(cuped_c, cuped_t, equal_var=False)

        st.markdown("#### 2. Significance Comparison")
        c_orig, c_cuped = st.columns(2)
        
        with c_orig:
            st.markdown("**Before CUPED (Standard T-Test)**")
            st.markdown(f"**P-Value:** {orig_pval:.4f}")
            st.markdown(_verdict_badge(orig_pval), unsafe_allow_html=True)

        with c_cuped:
            st.markdown("**After CUPED (Adjusted T-Test)**")
            st.markdown(f"**P-Value:** {cuped_pval:.4f}")
            st.markdown(_verdict_badge(cuped_pval), unsafe_allow_html=True)

def show():
    _inject_css()
    st.title("🧪 Experimentation Engine")
    st.caption("Phase 12 — A/B Testing, Power Analysis, and CUPED Variance Reduction")

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Standard A/B Test", "Power Analysis", "CUPED Variance Reduction"])

    with tab1:
        _standard_ab_test(df, num_cols, cat_cols)
        
    with tab2:
        _power_analysis()
        
    with tab3:
        _cuped_analysis(df, num_cols, cat_cols)