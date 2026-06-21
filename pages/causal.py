import streamlit as st
import pandas as pd
import numpy as np
from dowhy import CausalModel
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
      .step-header {{ color: {PRIMARY_COLOR}; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid {PRIMARY_COLOR}33; padding-bottom: 0.3rem; }}
    </style>
    """, unsafe_allow_html=True)

def _metric_card(col, value, label):
    col.markdown(f'<div class="metric-card"><div class="metric-val">{value}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

def show():
    _inject_css()
    st.title("🔗 Causal Inference Engine")
    st.caption("Phase 12 — Beyond Correlation: Estimating True Treatment Effects (powered by DoWhy)")

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded.")
        return

    st.info(
        "**Correlation vs. Causation:** This engine helps determine if a specific 'Treatment' "
        "(e.g., a marketing campaign, a new feature) actually *caused* a change in your 'Outcome' "
        "(e.g., sales, retention), while controlling for 'Confounders' (other variables that affect both)."
    )

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    st.markdown('<div class="step-header">1. Define the Causal Graph</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    treatment_col = c1.selectbox("Treatment (The Cause/Intervention):", options=cat_cols + num_cols, key="ci_treat")
    outcome_col = c2.selectbox("Outcome (The Effect):", options=num_cols, index=min(1, len(num_cols)-1) if len(num_cols) > 1 else 0, key="ci_out")

    if treatment_col == outcome_col:
        st.warning("Treatment and Outcome cannot be the same column.")
        return

    available_confounders = [c for c in all_cols if c not in [treatment_col, outcome_col]]
    confounder_cols = st.multiselect(
        "Confounders (Variables affecting BOTH Treatment and Outcome):",
        options=available_confounders,
        key="ci_conf"
    )

    if st.button("🚀 Estimate Causal Effect", type="primary"):
        if not confounder_cols:
            st.warning("While possible, omitting confounders assumes the treatment was perfectly randomized (like a strict A/B test). If this is observational data, please select known confounders.")
        
        # Clean data for modeling
        model_cols = [treatment_col, outcome_col] + confounder_cols
        clean_df = df[model_cols].dropna().copy()

        if len(clean_df) < 50:
            st.error(f"Not enough complete data points after dropping NaNs (Remaining: {len(clean_df)}). Causal inference requires larger sample sizes.")
            return

        # Ensure boolean/object treatments are handled (DoWhy prefers numeric or bool for binary treatments)
        if clean_df[treatment_col].dtype == 'object' and clean_df[treatment_col].nunique() == 2:
            clean_df[treatment_col] = clean_df[treatment_col] == clean_df[treatment_col].unique()[0]

        with st.spinner("Step 1/4: Building Causal Model..."):
            try:
                model = CausalModel(
                    data=clean_df,
                    treatment=treatment_col,
                    outcome=outcome_col,
                    common_causes=confounder_cols
                )
            except Exception as e:
                st.error(f"Failed to build model: {e}")
                return

        with st.spinner("Step 2/4: Identifying Causal Estimand..."):
            try:
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            except Exception as e:
                st.error(f"Failed to identify estimand: {e}")
                return

        with st.spinner("Step 3/4: Estimating the Effect (Linear Regression)..."):
            try:
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression",
                    test_significance=True
                )
                effect_value = estimate.value
                p_value = estimate.test_stat_significance().get('p_value', [np.nan])[0] if estimate.test_stat_significance() else np.nan
            except Exception as e:
                st.error(f"Failed to estimate effect: {e}")
                return

        with st.spinner("Step 4/4: Refuting the Estimate (Random Common Cause)..."):
            try:
                refutation = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="random_common_cause"
                )
                refutation_passed = "Yes" if abs(refutation.new_effect - effect_value) < (abs(effect_value) * 0.1) else "No (Fails Robustness)"
            except Exception:
                refutation_passed = "Could not compute"

        st.markdown('<div class="step-header">2. Causal Effect Results</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        _metric_card(m1, f"{effect_value:+.4f}", f"Estimated Causal Effect")
        _metric_card(m2, f"{p_value:.4f}" if not pd.isna(p_value) else "N/A", "P-Value")
        _metric_card(m3, refutation_passed, "Robust to Unobserved Confounders?")

        st.markdown("### 💬 Interpretation")
        direction = "increases" if effect_value > 0 else "decreases"
        st.success(
            f"**Finding:** Moving the **{treatment_col}** by one unit (or applying the treatment) "
            f"**{direction}** the **{outcome_col}** by an average of **{abs(effect_value):.4f}** units, "
            f"assuming the selected confounders ({', '.join(confounder_cols) if confounder_cols else 'None'}) "
            f"capture all major external influences."
        )

        with st.expander("🔍 View Technical Estimand details"):
            st.text(str(identified_estimand))