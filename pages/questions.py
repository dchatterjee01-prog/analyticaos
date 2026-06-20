# pages/questions.py

import streamlit as st
import pandas as pd
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR,
    TEXT_COLOR, ACCENT_COLOR
)


# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown(f"""
    <style>
      .aq-header {{
        font-size: 1.4rem; font-weight: 800;
        color: {PRIMARY_COLOR}; margin-bottom: 0.2rem;
      }}
      .aq-sub {{
        font-size: 0.82rem; color: {TEXT_COLOR}88;
        margin-bottom: 1.2rem;
      }}
      .metric-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {PRIMARY_COLOR}33;
        border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
      }}
      .metric-val {{
        font-size: 1.6rem; font-weight: 800; color: {ACCENT_COLOR};
      }}
      .metric-lbl {{
        font-size: 0.72rem; color: {TEXT_COLOR}88;
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem;
      }}
      .q-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {PRIMARY_COLOR}44;
        border-left: 4px solid {PRIMARY_COLOR};
        border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
      }}
      .q-card-stat  {{ border-left-color: {ACCENT_COLOR}; }}
      .q-card-pred  {{ border-left-color: #FF6B6B; }}
      .q-card-diag  {{ border-left-color: #FFD93D; }}
      .q-text {{
        font-size: 0.92rem; color: {TEXT_COLOR}; font-weight: 600;
      }}
      .q-meta {{ font-size: 0.74rem; color: {TEXT_COLOR}77; margin-top: 0.3rem; }}
      .badge {{
        display: inline-block; border-radius: 20px;
        padding: 0.15rem 0.65rem; font-size: 0.70rem;
        font-weight: 700; margin-right: 0.4rem;
      }}
      .badge-biz  {{ background:{PRIMARY_COLOR}22; border:1px solid {PRIMARY_COLOR}; color:{PRIMARY_COLOR}; }}
      .badge-stat {{ background:{ACCENT_COLOR}22;  border:1px solid {ACCENT_COLOR};  color:{ACCENT_COLOR}; }}
      .badge-pred {{ background:#FF6B6B22; border:1px solid #FF6B6B; color:#FF6B6B; }}
      .badge-diag {{ background:#FFD93D22; border:1px solid #FFD93D; color:#FFD93D; }}
      .info-box {{
        background: {SURFACE_COLOR};
        border-left: 3px solid {PRIMARY_COLOR};
        border-radius: 6px; padding: 0.8rem 1rem;
        font-size: 0.82rem; color: {TEXT_COLOR}CC;
        margin: 0.5rem 0 1rem 0;
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


# ── Silent profile scan ───────────────────────────────────────────────────────
def _run_profile_scan(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()

    date_cols = []
    for col in cat_cols:
        try:
            parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_cols.append(col)
        except Exception:
            pass

    id_cols = [
        c for c in cat_cols
        if df[c].nunique() / max(len(df), 1) > 0.9
        and c not in date_cols
    ]

    group_cols = [
        c for c in cat_cols
        if 2 <= df[c].nunique() <= 30
        and c not in date_cols
        and c not in id_cols
    ]

    continuous_cols = [c for c in numeric_cols if df[c].nunique() > 20]
    discrete_cols   = [c for c in numeric_cols if df[c].nunique() <= 20]

    missing           = df.isnull().sum()
    cols_with_missing = missing[missing > 0].index.tolist()

    st.session_state["aq_profile"] = {
        "numeric_cols":      numeric_cols,
        "cat_cols":          cat_cols,
        "date_cols":         date_cols,
        "group_cols":        group_cols,
        "id_cols":           id_cols,
        "continuous_cols":   continuous_cols,
        "discrete_cols":     discrete_cols,
        "cols_with_missing": cols_with_missing,
        "n_rows":            len(df),
        "n_cols":            len(df.columns)
    }


# ── Main entry ────────────────────────────────────────────────────────────────
def show():
    _inject_css()

    st.markdown(
        '<div class="aq-header">\U0001f9e0 Autonomous Question Generator</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="aq-sub">AI-powered analysis of your dataset — '
        'automatically surfaces the right questions to ask.</div>',
        unsafe_allow_html=True
    )

    df = st.session_state["df"].copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    _run_profile_scan(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001f4cb Dataset Profile",
        "\U0001f4a1 Question Bank",
        "\U0001f5fa\ufe0f Analysis Roadmap",
        "\U0001f4e4 Export Questions"
    ])

    with tab1:
        _dataset_profile(df)

    with tab2:
        _question_bank(df)

    with tab3:
        _analysis_roadmap(df)

    with tab4:
        _export_questions()


# ── Tab 1: Dataset Profile ────────────────────────────────────────────────────
def _dataset_profile(df):
    st.divider()
    st.markdown("### \U0001f52c Automated Dataset Intelligence")
    st.markdown(
        "<div class='info-box'>AnalyticaOS scans your dataset and builds a "
        "structural profile — the foundation for generating targeted "
        "analytical questions.</div>",
        unsafe_allow_html=True
    )

    p                 = st.session_state["aq_profile"]
    numeric_cols      = p["numeric_cols"]
    cat_cols          = p["cat_cols"]
    date_cols         = p["date_cols"]
    group_cols        = p["group_cols"]
    id_cols           = p["id_cols"]
    continuous_cols   = p["continuous_cols"]
    discrete_cols     = p["discrete_cols"]
    cols_with_missing = p["cols_with_missing"]
    bool_cols         = df.select_dtypes(include="bool").columns.tolist()
    missing           = df.isnull().sum()

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, f"{len(df):,}",          "Total Rows")
    _metric_card(m2, f"{len(df.columns)}",    "Total Columns")
    _metric_card(m3, f"{len(numeric_cols)}",  "Numeric Cols")
    _metric_card(m4, f"{len(cat_cols)}",      "Categorical Cols")

    st.markdown("")
    m5, m6, m7, m8 = st.columns(4)
    _metric_card(m5, f"{len(date_cols)}",         "Date-like Cols")
    _metric_card(m6, f"{len(group_cols)}",        "Grouping Cols")
    _metric_card(m7, f"{len(id_cols)}",           "ID/Key Cols")
    _metric_card(m8, f"{len(cols_with_missing)}", "Cols w/ Missing")

    st.divider()
    st.markdown("### \U0001f4ca Column Classification")

    rows = []
    for col in df.columns:
        if col in id_cols:
            role = "Identifier"
        elif col in date_cols:
            role = "Date/Time"
        elif col in group_cols:
            role = "Grouping"
        elif col in continuous_cols:
            role = "Continuous"
        elif col in discrete_cols:
            role = "Discrete"
        elif col in bool_cols:
            role = "Boolean"
        else:
            role = "Text"

        try:
            sample = str(df[col].dropna().iloc[0])[:40]
        except Exception:
            sample = "—"

        rows.append({
            "Column":   str(col),
            "Dtype":    str(df[col].dtype),
            "Role":     role,
            "Unique":   str(int(df[col].nunique())),
            "Missing%": f"{missing.get(col,0)/max(len(df),1)*100:.1f}%",
            "Sample":   sample
        })

    st.dataframe(pd.DataFrame(rows), width='stretch', height=350)
    st.success("\u2705 Dataset profile complete. Proceed to \U0001f4a1 Question Bank tab.")


# ── Tab 2: Question Bank ──────────────────────────────────────────────────────
def _question_bank(df):
    p               = st.session_state["aq_profile"]
    numeric_cols    = p["numeric_cols"]
    group_cols      = p["group_cols"]
    date_cols       = p["date_cols"]
    continuous_cols = p["continuous_cols"]
    cols_missing    = p["cols_with_missing"]

    st.divider()
    st.markdown("### \U0001f4a1 Auto-Generated Question Bank")
    st.markdown(
        "<div class='info-box'>Questions are generated by analyzing your "
        "dataset structure. Each question is tagged by type and linked to "
        "the relevant AnalyticaOS module.</div>",
        unsafe_allow_html=True
    )

    f1, f2 = st.columns([2, 1])
    with f1:
        filter_type = st.multiselect(
            "Filter by Question Type",
            options=["Business", "Statistical", "Predictive", "Diagnostic"],
            default=["Business", "Statistical", "Predictive", "Diagnostic"],
            key="aq_filter_type"
        )
    with f2:
        max_q = st.slider("Max Questions", min_value=5, max_value=50,
                          value=20, step=5, key="aq_max_q")

    st.markdown("")
    questions = []

    if "Business" in filter_type:
        for grp in group_cols[:3]:
            for met in continuous_cols[:3]:
                questions.append({"type": "Business",
                    "q": f"Which {grp} has the highest average {met}?",
                    "why": f"Compares mean {met} across {df[grp].nunique()} groups of {grp}.",
                    "module": "\U0001f4ca Pivot Table"})
                questions.append({"type": "Business",
                    "q": f"What is the total {met} contributed by each {grp}?",
                    "why": f"Pareto / contribution analysis across {grp}.",
                    "module": "\U0001f4ca Pivot Table \u2192 Contribution & Pareto"})
        for met in continuous_cols[:2]:
            questions.append({"type": "Business",
                "q": f"What are the top 10 records by {met}?",
                "why": f"Rankings reveal outlier performers in {met}.",
                "module": "\U0001f4ca Pivot Table \u2192 Top N Rankings"})
            questions.append({"type": "Business",
                "q": f"How is {met} distributed across the dataset?",
                "why": "Understanding spread, skew, and outliers.",
                "module": "\U0001f4c8 EDA \u2192 Distribution Analysis"})
        for dc in date_cols[:1]:
            for met in continuous_cols[:2]:
                questions.append({"type": "Business",
                    "q": f"How has {met} changed over time ({dc})?",
                    "why": f"Trend analysis of {met} using {dc} as timeline.",
                    "module": "\U0001f4ca Pivot Table \u2192 Time Intelligence"})

    if "Statistical" in filter_type:
        for grp in group_cols[:2]:
            n_groups = df[grp].nunique()
            if n_groups == 2:
                for met in continuous_cols[:2]:
                    questions.append({"type": "Statistical",
                        "q": f"Is there a significant difference in {met} between the two groups of {grp}?",
                        "why": f"{grp} has exactly 2 groups — ideal for a two-sample T-test.",
                        "module": "\U0001f4d0 Statistics \u2192 T-Test"})
            elif 3 <= n_groups <= 15:
                for met in continuous_cols[:2]:
                    questions.append({"type": "Statistical",
                        "q": f"Do the {n_groups} groups in {grp} have significantly different means of {met}?",
                        "why": f"ANOVA tests mean equality across {n_groups} groups.",
                        "module": "\U0001f4d0 Statistics \u2192 ANOVA"})
        for g1 in group_cols[:2]:
            for g2 in group_cols[:2]:
                if g1 != g2:
                    questions.append({"type": "Statistical",
                        "q": f"Is there a significant association between {g1} and {g2}?",
                        "why": "Chi-square test of independence on two categorical variables.",
                        "module": "\U0001f4d0 Statistics \u2192 Chi-Square"})
        for met in continuous_cols[:2]:
            questions.append({"type": "Statistical",
                "q": f"Is {met} normally distributed?",
                "why": "Normality affects choice of parametric vs non-parametric tests.",
                "module": "\U0001f4d0 Statistics \u2192 Normality Test"})
        if len(continuous_cols) >= 2:
            questions.append({"type": "Statistical",
                "q": "Which pairs of numeric variables are most strongly correlated?",
                "why": "Correlation matrix reveals multicollinearity and linear relationships.",
                "module": "\U0001f4c8 EDA \u2192 Correlation Analysis"})

    if "Predictive" in filter_type:
        for target in continuous_cols[:2]:
            other = [c for c in continuous_cols if c != target][:3]
            if other:
                questions.append({"type": "Predictive",
                    "q": f"Can we predict {target} using {', '.join(other)}?",
                    "why": f"Regression model with {len(other)} numeric predictors.",
                    "module": "\U0001f916 ML Engine \u2192 Regression"})
        for grp in group_cols[:2]:
            n_cls = df[grp].nunique()
            if 2 <= n_cls <= 10:
                questions.append({"type": "Predictive",
                    "q": f"Can we classify records into {grp} categories using numeric features?",
                    "why": f"{grp} has {n_cls} classes — suitable for classification.",
                    "module": "\U0001f916 ML Engine \u2192 Classification"})
        if date_cols and continuous_cols:
            for met in continuous_cols[:2]:
                questions.append({"type": "Predictive",
                    "q": f"What will {met} look like in the next period?",
                    "why": f"Time series forecasting using {date_cols[0]} as the time axis.",
                    "module": "\U0001f52e Forecasting"})

    if "Diagnostic" in filter_type:
        for met in continuous_cols[:3]:
            questions.append({"type": "Diagnostic",
                "q": f"Are there outliers in {met} that need investigation?",
                "why": "Outliers can distort aggregations and model training.",
                "module": "\U0001f4c8 EDA \u2192 Outlier Detection"})
        for col in cols_missing[:3]:
            questions.append({"type": "Diagnostic",
                "q": f"Why is {col} missing data, and how should it be handled?",
                "why": f"{col} has missing values that may bias results.",
                "module": "\U0001f9f9 Data Cleaning \u2192 Missing Value Analyzer"})
        if continuous_cols:
            questions.append({"type": "Diagnostic",
                "q": "Are there anomalous records that deviate significantly from the norm?",
                "why": "Anomaly detection flags fraud, errors, or rare events.",
                "module": "\U0001f6a8 Anomaly Detection"})
        if len(continuous_cols) >= 2:
            questions.append({"type": "Diagnostic",
                "q": "Are any features so highly correlated that they cause multicollinearity?",
                "why": "High VIF inflates regression coefficients.",
                "module": "\U0001f4c8 EDA \u2192 Correlation Analysis"})

    seen, unique_q = set(), []
    for q in questions:
        if q["q"] not in seen:
            seen.add(q["q"])
            unique_q.append(q)
    displayed = unique_q[:max_q]

    tc = {}
    for q in displayed:
        tc[q["type"]] = tc.get(q["type"], 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, str(len(displayed)), "Questions Generated")
    _metric_card(m2, str(tc.get("Business", 0)), "Business")
    _metric_card(m3, str(tc.get("Statistical", 0)), "Statistical")
    _metric_card(m4, str(tc.get("Predictive",0)+tc.get("Diagnostic",0)), "Predictive + Diagnostic")

    st.markdown("")
    st.divider()

    badge_map = {
        "Business":    ("badge-biz",  "Business"),
        "Statistical": ("badge-stat", "Statistical"),
        "Predictive":  ("badge-pred", "Predictive"),
        "Diagnostic":  ("badge-diag", "Diagnostic"),
    }
    card_class_map = {
        "Business":    "q-card",
        "Statistical": "q-card q-card-stat",
        "Predictive":  "q-card q-card-pred",
        "Diagnostic":  "q-card q-card-diag",
    }

    for i, q in enumerate(displayed, 1):
        bc, bl = badge_map[q["type"]]
        cc     = card_class_map[q["type"]]
        st.markdown(f"""
        <div class="{cc}">
          <span class="badge {bc}">{bl}</span>
          <span style="font-size:0.70rem;color:{TEXT_COLOR}55;">#{i} &middot; {q['module']}</span>
          <div class="q-text" style="margin-top:0.4rem;">{q['q']}</div>
          <div class="q-meta">&#128161; {q['why']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.session_state["aq_questions"] = displayed


# ── Tab 3: Analysis Roadmap ───────────────────────────────────────────────────
def _analysis_roadmap(df):
    p               = st.session_state["aq_profile"]
    numeric_cols    = p["numeric_cols"]
    group_cols      = p["group_cols"]
    date_cols       = p["date_cols"]
    continuous_cols = p["continuous_cols"]
    cols_missing    = p["cols_with_missing"]

    st.divider()
    st.markdown("### &#128506;&#65039; Recommended Analysis Roadmap")
    st.markdown(
        "<div class='info-box'>AnalyticaOS recommends a sequenced analytical "
        "workflow based on your dataset structure.</div>",
        unsafe_allow_html=True
    )

    roadmap = []
    step_n  = 1

    issues = []
    if cols_missing:
        issues.append(f"{len(cols_missing)} columns with missing values")
    if p["n_rows"] < 100:
        issues.append("small dataset (<100 rows)")

    roadmap.append({"step": step_n, "phase": "&#129529; Data Quality",
        "action": "Audit and clean the dataset before any analysis.",
        "detail": ("Issues detected: " + "; ".join(issues)) if issues else "No critical issues detected.",
        "module": "&#129529; Data Cleaning",
        "priority": "High" if issues else "Medium",
        "color": "#FF6B6B" if issues else ACCENT_COLOR})
    step_n += 1

    roadmap.append({"step": step_n, "phase": "&#128200; Exploratory Analysis",
        "action": "Understand distributions, correlations, and outliers.",
        "detail": f"{len(numeric_cols)} numeric columns available.",
        "module": "&#128200; EDA Engine", "priority": "High", "color": PRIMARY_COLOR})
    step_n += 1

    if group_cols or continuous_cols:
        roadmap.append({"step": step_n, "phase": "&#128202; Visual Analysis",
            "action": "Build charts to communicate patterns visually.",
            "detail": f"Recommended: bar charts across {group_cols[:2]}, scatter for {continuous_cols[:2]}.",
            "module": "&#128202; Visualization Engine", "priority": "Medium", "color": PRIMARY_COLOR})
        step_n += 1

    if group_cols and numeric_cols:
        roadmap.append({"step": step_n, "phase": "&#128202; Aggregation & Pivoting",
            "action": "Summarize metrics across key grouping dimensions.",
            "detail": f"Group by: {', '.join(group_cols[:3])}. Aggregate: {', '.join(numeric_cols[:3])}.",
            "module": "&#128202; Pivot Table Engine", "priority": "Medium", "color": ACCENT_COLOR})
        step_n += 1

    if date_cols:
        roadmap.append({"step": step_n, "phase": "&#128197; Time Series Analysis",
            "action": "Analyse trends, seasonality, and growth rates.",
            "detail": f"Date column detected: {date_cols[0]}.",
            "module": "&#128202; Pivot Table &#8594; Time Intelligence",
            "priority": "High", "color": "#FFD93D"})
        step_n += 1

    if group_cols and continuous_cols:
        n_grps = df[group_cols[0]].nunique() if group_cols else 0
        test   = "T-Test" if n_grps == 2 else ("ANOVA" if n_grps >= 3 else "Normality + Correlation")
        roadmap.append({"step": step_n, "phase": "&#128208; Statistical Testing",
            "action": f"Validate hypotheses using {test}.",
            "detail": f"Test group differences in {continuous_cols[0]}.",
            "module": f"&#128208; Statistics &#8594; {test}", "priority": "Medium", "color": ACCENT_COLOR})
        step_n += 1

    if len(numeric_cols) >= 2:
        roadmap.append({"step": step_n, "phase": "&#129302; Predictive Modelling",
            "action": "Train a model to predict a key target variable.",
            "detail": f"Candidate target: {continuous_cols[0] if continuous_cols else numeric_cols[0]}.",
            "module": "&#129302; ML Engine", "priority": "Medium", "color": PRIMARY_COLOR})
        step_n += 1

    if date_cols and continuous_cols:
        roadmap.append({"step": step_n, "phase": "&#128302; Forecasting",
            "action": f"Forecast future values of {continuous_cols[0]}.",
            "detail": "Use Forecasting Engine with detected date column.",
            "module": "&#128302; Forecasting Engine", "priority": "Low", "color": "#888"})
        step_n += 1

    roadmap.append({"step": step_n, "phase": "&#127963;&#65039; Executive Summary",
        "action": "Synthesize findings into a strategy-ready report.",
        "detail": "Auto-generate an executive brief with key insights and recommendations.",
        "module": "&#127963;&#65039; Executive Console (Phase 10)", "priority": "Low", "color": "#888"})

    high   = sum(1 for r in roadmap if r["priority"] == "High")
    medium = sum(1 for r in roadmap if r["priority"] == "Medium")
    low    = sum(1 for r in roadmap if r["priority"] == "Low")

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, str(len(roadmap)), "Total Steps")
    _metric_card(m2, str(high),         "High Priority")
    _metric_card(m3, str(medium),       "Medium Priority")
    _metric_card(m4, str(low),          "Low / Future")

    st.markdown("")
    st.divider()

    p_badge = {
        "High":   f"<span style='background:#FF6B6B22;border:1px solid #FF6B6B;color:#FF6B6B;border-radius:20px;padding:0.15rem 0.6rem;font-size:0.70rem;font-weight:700;'>&#128308; High</span>",
        "Medium": f"<span style='background:{ACCENT_COLOR}22;border:1px solid {ACCENT_COLOR};color:{ACCENT_COLOR};border-radius:20px;padding:0.15rem 0.6rem;font-size:0.70rem;font-weight:700;'>&#127937; Medium</span>",
        "Low":    f"<span style='background:#88888822;border:1px solid #888;color:#888;border-radius:20px;padding:0.15rem 0.6rem;font-size:0.70rem;font-weight:700;'>&#9898; Low</span>",
    }

    for r in roadmap:
        st.markdown(f"""
        <div style="background:{SURFACE_COLOR};border:1px solid {r['color']}44;
                    border-left:4px solid {r['color']};border-radius:8px;
                    padding:0.9rem 1.1rem;margin-bottom:0.7rem;">
          <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;">
            <span style="background:{r['color']}22;border:1px solid {r['color']};
                         color:{r['color']};border-radius:50%;width:1.6rem;height:1.6rem;
                         display:inline-flex;align-items:center;justify-content:center;
                         font-size:0.75rem;font-weight:800;">{r['step']}</span>
            <span style="font-size:0.95rem;font-weight:700;color:{TEXT_COLOR};">{r['phase']}</span>
            {p_badge[r['priority']]}
          </div>
          <div style="font-size:0.86rem;color:{TEXT_COLOR};margin-bottom:0.2rem;">{r['action']}</div>
          <div style="font-size:0.75rem;color:{TEXT_COLOR}77;">
            &#128206; {r['detail']}<br>
            <span style="color:{r['color']};font-weight:600;">&#8594; {r['module']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.session_state["aq_roadmap"] = roadmap


# ── Tab 4: Export Questions ───────────────────────────────────────────────────
def _export_questions():
    import json

    if "aq_questions" not in st.session_state:
        st.info("&#128070; Visit &#128161; Question Bank tab first.", icon="&#9888;&#65039;")
        return

    questions = st.session_state["aq_questions"]
    roadmap   = st.session_state.get("aq_roadmap", [])
    profile   = st.session_state.get("aq_profile", {})

    st.divider()
    st.markdown("### &#128228; Export Your Analysis Plan")
    st.markdown(
        "<div class='info-box'>Download your auto-generated question bank "
        "and analysis roadmap as CSV, JSON, or a Markdown brief.</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, str(len(questions)),            "Questions")
    _metric_card(m2, str(len(roadmap)),              "Roadmap Steps")
    _metric_card(m3, str(profile.get("n_rows","—")), "Dataset Rows")
    _metric_card(m4, str(profile.get("n_cols","—")), "Dataset Cols")

    st.markdown("")
    st.divider()

    q_df = pd.DataFrame([{
        "No": i+1, "Type": q["type"],
        "Question": q["q"], "Why": q["why"], "Module": q["module"]
    } for i, q in enumerate(questions)])

    r_df = pd.DataFrame([{
        "Step": r["step"], "Phase": r["phase"],
        "Action": r["action"], "Detail": r["detail"],
        "Module": r["module"], "Priority": r["priority"]
    } for r in roadmap])

    def _build_md():
        lines = [
            "# AnalyticaOS — Autonomous Analysis Brief", "",
            "## Dataset Profile",
            f"- **Rows:** {profile.get('n_rows','—')}",
            f"- **Columns:** {profile.get('n_cols','—')}",
            "", "---", "", "## Question Bank", ""
        ]
        for qtype in ["Business", "Statistical", "Predictive", "Diagnostic"]:
            typed = [q for q in questions if q["type"] == qtype]
            if not typed:
                continue
            lines.append(f"### {qtype}")
            for i, q in enumerate(typed, 1):
                lines.append(f"{i}. **{q['q']}**")
                lines.append(f"   - *{q['why']}*")
                lines.append(f"   - -> {q['module']}")
            lines.append("")
        lines += ["---", "", "## Analysis Roadmap", ""]
        for r in roadmap:
            lines.append(f"**Step {r['step']} - {r['phase']}** [{r['priority']} Priority]")
            lines.append(f"- {r['action']}")
            lines.append(f"- -> {r['module']}")
            lines.append("")
        lines.append("*Generated by AnalyticaOS*")
        return "\n".join(lines)

    md_text = _build_md()

    st.markdown("#### &#128229; Download Options")
    d1, d2, d3 = st.columns(3)

    d1.download_button("&#11015;&#65039; Questions CSV",
        data=q_df.to_csv(index=False).encode("utf-8"),
        file_name="analyticaos_questions.csv", mime="text/csv",
        width='stretch', key="aq_dl_csv")

    d2.download_button("&#11015;&#65039; Full JSON Package",
        data=json.dumps({"profile": profile, "questions": questions,
                         "roadmap": roadmap}, indent=2).encode("utf-8"),
        file_name="analyticaos_analysis_plan.json", mime="application/json",
        width='stretch', key="aq_dl_json")

    d3.download_button("&#11015;&#65039; Markdown Brief",
        data=md_text.encode("utf-8"),
        file_name="analyticaos_brief.md", mime="text/markdown",
        width='stretch', key="aq_dl_md")

    st.divider()
    st.markdown("#### Question Bank Preview")
    st.dataframe(q_df.astype(str), width='stretch', height=300)

    st.markdown("")
    st.markdown("#### Roadmap Preview")
    st.dataframe(r_df.astype(str), width='stretch', height=280)

    st.divider()
    with st.expander("Preview Markdown Brief"):
        st.markdown(md_text)
