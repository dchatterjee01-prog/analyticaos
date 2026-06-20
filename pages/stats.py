# pages/stats.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR,
    TEXT_COLOR, ACCENT_COLOR, SUCCESS_BG, SUCCESS_COLOR, WARNING_BG, WARNING_COLOR
)


def _inject_css():
    st.markdown(f"""
    <style>
      .stat-header {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.2rem;
      }}
      .stat-sub {{
        font-size: 0.82rem;
        color: {TEXT_COLOR}88;
        margin-bottom: 1.2rem;
      }}
      .metric-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {PRIMARY_COLOR}33;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
      }}
      .metric-val {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {ACCENT_COLOR};
      }}
      .metric-lbl {{
        font-size: 0.72rem;
        color: {TEXT_COLOR}88;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
      }}
      .verdict-sig {{
        background: {SUCCESS_BG};
    color: {SUCCESS_COLOR};
    border: 1px solid {SUCCESS_COLOR};
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
      }}
      .verdict-nosig {{
        background: {WARNING_BG};
    color: {WARNING_COLOR};
    border: 1px solid {WARNING_COLOR};
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
      }}
    </style>
    """, unsafe_allow_html=True)


def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{value}</div>
      <div class="metric-lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _verdict_badge(p_value, alpha=0.05):
    if p_value < alpha:
        return (
            f'<span class="verdict-sig">✅ Statistically Significant '
            f'(p={p_value:.4f} &lt; {alpha})</span>'
        )
    return (
        f'<span class="verdict-nosig">⚪ Not Significant '
        f'(p={p_value:.4f} &ge; {alpha})</span>'
    )


# ── Phase 11 prerequisite: persist test results for Report Generator ─────
def _save_stat_result(test_type, variables, statistic_name, statistic_value,
                       p_value, interpretation, alpha=0.05):
    """
    Appends one hypothesis-test result to st.session_state["stats_results"].
    Schema (one dict per test run):
      {
        "test_type": str          e.g. "T-Test (Two-Sample)"
        "variables": str          e.g. "Sales by Region (North vs South)"
        "statistic_name": str     e.g. "T-Statistic"
        "statistic_value": float
        "p_value": float
        "significant": bool       p_value < alpha
        "interpretation": str     plain-language verdict sentence
      }
    Report Generator reads st.session_state.get("stats_results", []) to
    build the Statistical Findings section of the .docx report.
    """
    if "stats_results" not in st.session_state:
        st.session_state["stats_results"] = []

    st.session_state["stats_results"].append({
        "test_type": test_type,
        "variables": variables,
        "statistic_name": statistic_name,
        "statistic_value": round(float(statistic_value), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha),
        "interpretation": interpretation
    })


def show():
    _inject_css()

    st.markdown('<div class="stat-header">📐 Statistical Engine</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="stat-sub">Run hypothesis tests and get '
                'plain-language interpretation — no stats degree required.</div>',
                unsafe_allow_html=True)

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        st.error("No numeric columns found. Use Data Cleaning → "
                 "Data Type Fixer first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 T-Test", "📊 Chi-Square", "📈 ANOVA", "🔔 Normality Test"
    ])

    with tab1:
        _t_test(df, num_cols, cat_cols)

    with tab2:
        _chi_square(df, cat_cols)

    with tab3:
        _anova(df, num_cols, cat_cols)

    with tab4:
        _normality_test(df, num_cols)


# ── Tab 1: T-Test ──────────────────────────────────────────────────────
def _t_test(df, num_cols, cat_cols):
    st.divider()
    st.markdown("### ⚙️ Configure T-Test")

    test_type = st.radio(
        "Test type:",
        options=["Two-Sample (compare groups)", "One-Sample (vs fixed value)"],
        horizontal=True,
        key="tt_type"
    )

    if test_type == "Two-Sample (compare groups)":
        if not cat_cols:
            st.info("Need at least 1 categorical column to split into groups.")
            return

        c1, c2 = st.columns(2)
        group_col = c1.selectbox("Group column:", options=cat_cols, key="tt_group")
        value_col = c2.selectbox("Value column (numeric):", options=num_cols,
                                 key="tt_value")

        groups = df[group_col].dropna().unique().tolist()
        if len(groups) < 2:
            st.warning(f"'{group_col}' has fewer than 2 unique groups.")
            return

        g1, g2 = st.columns(2)
        group_a = g1.selectbox("Group A:", options=groups, index=0, key="tt_a")
        group_b = g2.selectbox("Group B:", options=groups,
                               index=min(1, len(groups)-1), key="tt_b")

        if group_a == group_b:
            st.warning("Please select two different groups.")
            return

        data_a = df[df[group_col] == group_a][value_col].dropna()
        data_b = df[df[group_col] == group_b][value_col].dropna()

        if len(data_a) < 2 or len(data_b) < 2:
            st.error("Each group needs at least 2 data points.")
            return

        if st.button("🧪 Run T-Test", key="run_tt2"):
            t_stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            _metric_card(m1, f"{data_a.mean():.3f}", f"Mean — {group_a}")
            _metric_card(m2, f"{data_b.mean():.3f}", f"Mean — {group_b}")
            _metric_card(m3, f"{t_stat:.3f}",        "T-Statistic")
            _metric_card(m4, f"{p_val:.4f}",         "P-Value")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(_verdict_badge(p_val), unsafe_allow_html=True)

            st.markdown("### 💬 Interpretation")
            if p_val < 0.05:
                interp = (
                    f"There IS a statistically significant difference in "
                    f"{value_col} between {group_a} "
                    f"(mean={data_a.mean():.2f}) and {group_b} "
                    f"(mean={data_b.mean():.2f}). This difference is unlikely "
                    f"due to random chance (p={p_val:.4f})."
                )
                st.success(interp)
            else:
                interp = (
                    f"There is NO statistically significant difference in "
                    f"{value_col} between {group_a} and {group_b} "
                    f"(p={p_val:.4f}). Any observed difference could be due "
                    f"to random chance."
                )
                st.info(interp)

            _save_stat_result(
                test_type="T-Test (Two-Sample)",
                variables=f"{value_col} by {group_col} ({group_a} vs {group_b})",
                statistic_name="T-Statistic",
                statistic_value=t_stat,
                p_value=p_val,
                interpretation=interp
            )

            st.markdown("### 📊 Distribution Comparison")
            fig = go.Figure()
            fig.add_trace(go.Box(y=data_a, name=str(group_a),
                                 marker_color=PRIMARY_COLOR, boxmean=True))
            fig.add_trace(go.Box(y=data_b, name=str(group_b),
                                 marker_color=ACCENT_COLOR, boxmean=True))
            fig.update_layout(
                paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR, title=f"{value_col} — {group_a} vs {group_b}",
                title_font_color=PRIMARY_COLOR,
                yaxis=dict(gridcolor="#333"),
                margin=dict(t=60, b=40), height=400
            )
            st.plotly_chart(fig, width="stretch")

    else:
        c1, c2 = st.columns(2)
        value_col = c1.selectbox("Value column:", options=num_cols, key="tt1_val")
        test_value = c2.number_input("Compare against value:", value=0.0,
                                     key="tt1_against")

        data = df[value_col].dropna()

        if st.button("🧪 Run One-Sample T-Test", key="run_tt1"):
            t_stat, p_val = stats.ttest_1samp(data, test_value)

            st.divider()
            m1, m2, m3 = st.columns(3)
            _metric_card(m1, f"{data.mean():.3f}", "Sample Mean")
            _metric_card(m2, f"{t_stat:.3f}",      "T-Statistic")
            _metric_card(m3, f"{p_val:.4f}",       "P-Value")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(_verdict_badge(p_val), unsafe_allow_html=True)

            st.markdown("### 💬 Interpretation")
            if p_val < 0.05:
                interp = (
                    f"The mean of {value_col} ({data.mean():.2f}) is "
                    f"significantly different from {test_value} (p={p_val:.4f})."
                )
                st.success(interp)
            else:
                interp = (
                    f"The mean of {value_col} ({data.mean():.2f}) is NOT "
                    f"significantly different from {test_value} (p={p_val:.4f})."
                )
                st.info(interp)

            _save_stat_result(
                test_type="T-Test (One-Sample)",
                variables=f"{value_col} vs fixed value {test_value}",
                statistic_name="T-Statistic",
                statistic_value=t_stat,
                p_value=p_val,
                interpretation=interp
            )


# ── Tab 2: Chi-Square Test ─────────────────────────────────────────────
def _chi_square(df, cat_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Chi-Square Test")

    if len(cat_cols) < 2:
        st.info("Need at least 2 categorical columns to test independence.")
        return

    c1, c2 = st.columns(2)
    col_a = c1.selectbox("Variable A:", options=cat_cols, key="chi_a")
    col_b = c2.selectbox(
        "Variable B:",
        options=[c for c in cat_cols if c != col_a] or cat_cols,
        key="chi_b"
    )

    if st.button("📊 Run Chi-Square Test", key="run_chi"):
        contingency = pd.crosstab(df[col_a], df[col_b])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            st.error("Need at least 2 categories in each variable.")
            return

        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

        st.divider()
        m1, m2, m3 = st.columns(3)
        _metric_card(m1, f"{chi2:.3f}", "Chi-Square Statistic")
        _metric_card(m2, f"{dof}",      "Degrees of Freedom")
        _metric_card(m3, f"{p_val:.4f}", "P-Value")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_verdict_badge(p_val), unsafe_allow_html=True)

        st.markdown("### 💬 Interpretation")
        if p_val < 0.05:
            interp = (
                f"There IS a statistically significant association between "
                f"{col_a} and {col_b} (p={p_val:.4f}). Knowing one "
                f"variable tells you something about the other."
            )
            st.success(interp)
        else:
            interp = (
                f"There is NO statistically significant association between "
                f"{col_a} and {col_b} (p={p_val:.4f}). These variables "
                f"appear to be independent."
            )
            st.info(interp)

        _save_stat_result(
            test_type="Chi-Square Test of Independence",
            variables=f"{col_a} vs {col_b}",
            statistic_name="Chi-Square Statistic",
            statistic_value=chi2,
            p_value=p_val,
            interpretation=interp
        )

        st.markdown("### 📋 Contingency Table (Observed Counts)")
        st.dataframe(contingency.astype(str), width="stretch")

        st.markdown("### 🌡️ Heatmap of Observed Counts")
        fig = go.Figure(go.Heatmap(
            z=contingency.values,
            x=contingency.columns.astype(str).tolist(),
            y=contingency.index.astype(str).tolist(),
            colorscale="Viridis",
            text=contingency.values,
            texttemplate="%{text}",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title=f"{col_a} × {col_b} — Observed Counts",
            title_font_color=PRIMARY_COLOR,
            margin=dict(t=60, b=60, l=100), height=420
        )
        st.plotly_chart(fig, width="stretch")


# ── Tab 3: ANOVA ───────────────────────────────────────────────────────
def _anova(df, num_cols, cat_cols):
    st.divider()
    st.markdown("### ⚙️ Configure One-Way ANOVA")

    if not cat_cols:
        st.info("Need at least 1 categorical column to define groups.")
        return

    c1, c2 = st.columns(2)
    group_col = c1.selectbox("Group column:", options=cat_cols, key="an_group")
    value_col = c2.selectbox("Value column (numeric):", options=num_cols,
                             key="an_value")

    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 3:
        st.warning(
            f"'{group_col}' has only {len(groups)} groups. ANOVA needs 3+ "
            f"groups — use T-Test for 2 groups."
        )
        return

    group_data = [
        df[df[group_col] == g][value_col].dropna()
        for g in groups
    ]
    group_data = [g for g in group_data if len(g) >= 2]

    if len(group_data) < 3:
        st.error("Not enough data in groups (need 2+ values per group).")
        return

    if st.button("📈 Run ANOVA", key="run_anova"):
        f_stat, p_val = stats.f_oneway(*group_data)

        st.divider()
        m1, m2, m3 = st.columns(3)
        _metric_card(m1, f"{f_stat:.3f}", "F-Statistic")
        _metric_card(m2, f"{len(groups)}", "Groups Compared")
        _metric_card(m3, f"{p_val:.4f}",  "P-Value")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_verdict_badge(p_val), unsafe_allow_html=True)

        st.markdown("### 💬 Interpretation")
        if p_val < 0.05:
            interp = (
                f"There IS a statistically significant difference in "
                f"{value_col} across the {group_col} groups "
                f"(p={p_val:.4f}). At least one group differs from the others."
            )
            st.success(interp)
        else:
            interp = (
                f"There is NO statistically significant difference in "
                f"{value_col} across the {group_col} groups "
                f"(p={p_val:.4f})."
            )
            st.info(interp)

        _save_stat_result(
            test_type="One-Way ANOVA",
            variables=f"{value_col} across {group_col} ({len(groups)} groups)",
            statistic_name="F-Statistic",
            statistic_value=f_stat,
            p_value=p_val,
            interpretation=interp
        )

        st.markdown("### 📊 Group Means Comparison")
        means_df = (
            df.groupby(group_col)[value_col]
            .agg(["mean", "std", "count"])
            .reset_index()
            .round(3)
        )
        means_df.columns = [group_col, "Mean", "Std Dev", "Count"]
        st.dataframe(means_df, width="stretch")

        fig = go.Figure()
        for g, gd in zip(groups, [df[df[group_col] == g][value_col].dropna()
                                  for g in groups]):
            fig.add_trace(go.Box(y=gd, name=str(g), boxmean=True))
        fig.update_layout(
            paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title=f"{value_col} by {group_col}",
            title_font_color=PRIMARY_COLOR,
            yaxis=dict(gridcolor="#333"),
            margin=dict(t=60, b=40), height=420,
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch")


# ── Tab 4: Normality Test ──────────────────────────────────────────────
def _normality_test(df, num_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Normality Test")

    st.info(
        "Checks whether a numeric column follows a normal (bell curve) "
        "distribution — important before running t-tests, ANOVA, or "
        "linear regression."
    )

    value_col = st.selectbox("Column to test:", options=num_cols,
                             key="norm_col")

    data = df[value_col].dropna()

    if len(data) < 3:
        st.error("Need at least 3 data points.")
        return

    if len(data) > 5000:
        st.warning(
            f"Shapiro-Wilk test works best under 5000 samples. "
            f"Using a random sample of 5000 from {len(data)} rows."
        )
        data = data.sample(5000, random_state=42)

    if st.button("🔔 Run Normality Test", key="run_norm"):
        shapiro_stat, shapiro_p = stats.shapiro(data)
        skewness = data.skew()
        kurtosis = data.kurt()

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        _metric_card(m1, f"{shapiro_stat:.4f}", "Shapiro-Wilk Statistic")
        _metric_card(m2, f"{shapiro_p:.4f}",    "P-Value")
        _metric_card(m3, f"{skewness:.3f}",     "Skewness")
        _metric_card(m4, f"{kurtosis:.3f}",     "Kurtosis")

        st.markdown("<br>", unsafe_allow_html=True)

        if shapiro_p < 0.05:
            st.markdown(
                '<span class="verdict-nosig">⚠️ NOT Normally Distributed '
                f'(p={shapiro_p:.4f} &lt; 0.05)</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="verdict-sig">✅ Normally Distributed '
                f'(p={shapiro_p:.4f} &ge; 0.05)</span>',
                unsafe_allow_html=True
            )

        st.markdown("### 💬 Interpretation")
        if shapiro_p < 0.05:
            interp = (
                f"{value_col} does NOT follow a normal distribution "
                f"(p={shapiro_p:.4f}). Consider using non-parametric tests "
                f"(e.g. Mann-Whitney U instead of t-test) or transforming "
                f"the data (log, square root) before parametric analysis."
            )
            st.warning(interp)
        else:
            interp = (
                f"{value_col} appears to follow a normal distribution "
                f"(p={shapiro_p:.4f}). Parametric tests like t-test, ANOVA, "
                f"and linear regression are appropriate to use."
            )
            st.success(interp)

        _save_stat_result(
            test_type="Normality Test (Shapiro-Wilk)",
            variables=value_col,
            statistic_name="Shapiro-Wilk Statistic",
            statistic_value=shapiro_stat,
            p_value=shapiro_p,
            interpretation=interp
        )

        skew_note = (
            "right-skewed (long tail to the right)" if skewness > 0.5 else
            "left-skewed (long tail to the left)" if skewness < -0.5 else
            "fairly symmetric"
        )
        st.markdown(f"📈 Distribution shape: **{skew_note}** (skewness={skewness:.2f})")

        # ── Histogram with normal curve overlay ──
        st.markdown("### 📊 Histogram vs Normal Curve")
        mean, std = data.mean(), data.std()
        x_range = np.linspace(data.min(), data.max(), 200)
        normal_curve = stats.norm.pdf(x_range, mean, std)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=data, histnorm="probability density",
            marker_color=PRIMARY_COLOR, opacity=0.7,
            name="Observed Data"
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=normal_curve,
            mode="lines", line=dict(color=ACCENT_COLOR, width=2.5),
            name="Normal Curve (theoretical)"
        ))
        fig_hist.update_layout(
            paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title=f"{value_col} — Distribution vs Normal",
            title_font_color=PRIMARY_COLOR,
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#333"),
            margin=dict(t=60, b=40), height=400
        )
        st.plotly_chart(fig_hist, width="stretch")

        # ── Q-Q Plot ──
        st.markdown("### 📈 Q-Q Plot (Quantile-Quantile)")
        st.caption(
            "Points following the diagonal red line indicate normal "
            "distribution. Deviation from the line indicates non-normality."
        )

        osm, osr = stats.probplot(data, dist="norm", fit=False)
        slope, intercept, r = stats.linregress(osm, osr)[:3]
        line_x = np.array([osm.min(), osm.max()])
        line_y = slope * line_x + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(color=PRIMARY_COLOR, size=5, opacity=0.6),
            name="Sample Quantiles"
        ))
        fig_qq.add_trace(go.Scatter(
            x=line_x, y=line_y, mode="lines",
            line=dict(color="#FF6B6B", width=2, dash="dash"),
            name="Reference Line"
        ))
        fig_qq.update_layout(
            paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title=f"Q-Q Plot — {value_col}",
            title_font_color=PRIMARY_COLOR,
            xaxis=dict(title="Theoretical Quantiles", gridcolor="#333"),
            yaxis=dict(title="Sample Quantiles", gridcolor="#333"),
            legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#333"),
            margin=dict(t=60, b=40), height=420
        )
        st.plotly_chart(fig_qq, width="stretch")