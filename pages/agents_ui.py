"""
agents_ui.py — Phase 9: Multi-Agent System UI
Lets the user run the full agent pipeline (DataQuality -> Insight -> Modeling)
with one click and view results in the existing dark-theme card style.
"""
import streamlit as st
import pandas as pd
from agents import Orchestrator

PRIMARY_COLOR    = "#6C63FF"
BACKGROUND_COLOR = "#0F0F1A"
SURFACE_COLOR    = "#1A1A2E"
TEXT_COLOR       = "#E8E8F0"
ACCENT_COLOR     = "#00D4AA"

SEVERITY_COLORS = {
    "low":    ACCENT_COLOR,
    "medium": "#FFB020",
    "high":   "#FF5C5C",
}

STATUS_BADGE = {
    "ok":      ("✅ OK",      ACCENT_COLOR),
    "warning": ("⚠️ Warning", "#FFB020"),
    "error":   ("❌ Error",   "#FF5C5C"),
}


def _inject_css():
    st.markdown(f"""
    <style>
    .agents-header {{
        font-size: 2rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.2rem;
    }}
    .agents-sub {{
        color: {TEXT_COLOR};
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background-color: {SURFACE_COLOR};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid rgba(108,99,255,0.25);
    }}
    .metric-val {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}
    .metric-lbl {{
        font-size: 0.85rem;
        color: {TEXT_COLOR};
        opacity: 0.7;
    }}
    .finding-card {{
        background-color: {SURFACE_COLOR};
        border-left: 4px solid var(--sev-color);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
    }}
    .finding-title {{
        font-weight: 600;
        color: {TEXT_COLOR};
    }}
    .finding-detail {{
        font-size: 0.88rem;
        color: {TEXT_COLOR};
        opacity: 0.75;
    }}
    .rec-pill {{
        display: inline-block;
        background-color: rgba(0,212,170,0.12);
        color: {ACCENT_COLOR};
        border: 1px solid rgba(0,212,170,0.35);
        border-radius: 20px;
        padding: 0.35rem 0.9rem;
        margin: 0.25rem 0.25rem 0.25rem 0;
        font-size: 0.85rem;
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


def _render_finding(f: dict):
    sev = f.get("severity", "low")
    color = SEVERITY_COLORS.get(sev, ACCENT_COLOR)
    st.markdown(f"""
        <div class="finding-card" style="--sev-color:{color};">
            <div class="finding-title">{f['title']}</div>
            <div class="finding-detail">{f['detail']}</div>
        </div>
    """, unsafe_allow_html=True)


def _render_agent_tab(agent_name: str, result):
    badge_text, badge_color = STATUS_BADGE.get(result.status, ("Unknown", TEXT_COLOR))
    st.markdown(
        f"<span style='color:{badge_color}; font-weight:600;'>{badge_text}</span>",
        unsafe_allow_html=True
    )
    st.write(result.summary)
    st.divider()

    if result.findings:
        st.markdown("**Findings**")
        for f in result.findings:
            _render_finding(f)
    else:
        st.caption("No findings reported.")

    # Agent-specific artifact previews
    artifacts = result.artifacts or {}

    if agent_name == "DataQualityAgent":
        if "missing_summary" in artifacts and not artifacts["missing_summary"].empty:
            st.markdown("**Missing Value Summary**")
            st.dataframe(artifacts["missing_summary"], width="stretch")

    if agent_name == "InsightAgent":
        if "descriptive_stats" in artifacts:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(artifacts["descriptive_stats"], width="stretch")
        if artifacts.get("strong_correlations"):
            st.markdown("**Strong Correlations**")
            corr_df = pd.DataFrame(
                artifacts["strong_correlations"],
                columns=["Column A", "Column B", "Correlation"]
            )
            st.dataframe(corr_df, width="stretch")

    if agent_name == "ModelingAgent":
        if "metrics" in artifacts:
            st.markdown("**Model Metrics**")
            cols = st.columns(len(artifacts["metrics"]))
            for c, (k, v) in zip(cols, artifacts["metrics"].items()):
                _metric_card(c, v, k.upper())
        if "feature_importance" in artifacts:
            st.markdown("**Feature Importance**")
            st.dataframe(artifacts["feature_importance"], width="stretch")


def show():
    _inject_css()
    st.markdown('<div class="agents-header">🤖 Multi-Agent System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="agents-sub">Run the autonomous agent pipeline: '
        'Data Quality → Insight → Modeling</div>',
        unsafe_allow_html=True
    )

    df = st.session_state.get("df")

    with st.expander("⚙️ Pipeline Options", expanded=False):
        target_col = st.selectbox(
            "Target column for Modeling Agent (optional)",
            options=["(auto-detect)"] + df.columns.tolist(),
        )

    run_clicked = st.button("▶️ Run Multi-Agent Analysis", type="primary", width="stretch")

    if run_clicked:
        context = {}
        if "aq_roadmap" in st.session_state:
            context["aq_roadmap"] = st.session_state["aq_roadmap"]
        if target_col != "(auto-detect)":
            context["target_col"] = target_col

        with st.spinner("Agents working..."):
            report = Orchestrator().run(df, context=context)

        st.session_state["agent_report"] = report

    if "agent_report" not in st.session_state:
        st.info("Click **Run Multi-Agent Analysis** to start.")
        return

    report = st.session_state["agent_report"]

    # ── Overall status row ──────────────────────────────────────────────
    badge_text, badge_color = STATUS_BADGE.get(report["overall_status"], ("Unknown", TEXT_COLOR))
    st.markdown(
        f"### Overall Status: <span style='color:{badge_color};'>{badge_text}</span>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    _metric_card(c1, len(report["pipeline"]), "Agents Run")
    _metric_card(c2, len(report["all_findings"]), "Total Findings")
    _metric_card(c3, len(report["all_recommendations"]), "Recommendations")

    st.divider()

    # ── Recommendations ─────────────────────────────────────────────────
    if report["all_recommendations"]:
        st.markdown("**🎯 Top Recommendations**")
        pills_html = "".join(
            f"<span class='rec-pill'>{r}</span>" for r in report["all_recommendations"]
        )
        st.markdown(pills_html, unsafe_allow_html=True)
        st.divider()

    # ── Per-agent tabs ───────────────────────────────────────────────────
    tab_names = report["pipeline"]
    tabs = st.tabs([f"🔹 {n}" for n in tab_names])
    for tab, name in zip(tabs, tab_names):
        with tab:
            result = report["results"].get(name)
            if result is None:
                st.warning(f"{name} did not produce a result.")
                continue
            _render_agent_tab(name, result)