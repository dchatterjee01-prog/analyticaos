# pages/report.py
import streamlit as st
import pandas as pd
from datetime import datetime
from google import genai
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR, TEXT_COLOR,
    BORDER_COLOR, MUTED_COLOR, SUCCESS_COLOR, SUCCESS_BG,
    WARNING_COLOR, WARNING_BG, ERROR_COLOR, ERROR_BG
)
from pages.executive import LocalExecutiveEngine
from pages.report_builder import build_docx_report


def _inject_css():
    st.markdown(f"""
        <style>
        .report-header {{ color: {PRIMARY_COLOR}; font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
        .report-sub {{ color: {MUTED_COLOR}; font-size: 14px; margin-bottom: 24px; }}
        .metric-card {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; padding: 16px; border-radius: 6px; text-align: center; }}
        .metric-val {{ color: {TEXT_COLOR}; font-size: 24px; font-weight: 700; }}
        .metric-lbl {{ color: {MUTED_COLOR}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
        .section-row {{
            display: flex; align-items: center; justify-content: space-between;
            background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR};
            border-radius: 6px; padding: 12px 16px; margin-bottom: 8px;
        }}
        .section-name {{ color: {TEXT_COLOR}; font-size: 14px; font-weight: 600; }}
        .section-desc {{ color: {MUTED_COLOR}; font-size: 12px; margin-top: 2px; }}
        .badge-ready {{
            background-color: {SUCCESS_BG}; color: {SUCCESS_COLOR};
            border: 1px solid {SUCCESS_COLOR}; border-radius: 20px;
            padding: 3px 12px; font-size: 11px; font-weight: 700;
        }}
        .badge-missing {{
            background-color: {WARNING_BG}; color: {WARNING_COLOR};
            border: 1px solid {WARNING_COLOR}; border-radius: 20px;
            padding: 3px 12px; font-size: 11px; font-weight: 700;
        }}
        .narrative-box {{
            background: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR};
            border-left: 4px solid {PRIMARY_COLOR}; border-radius: 8px;
            padding: 1.5rem; margin-top: 0.5rem; font-size: 1.05rem; line-height: 1.6;
        }}
        </style>
    """, unsafe_allow_html=True)


def _metric_card(col, value, label):
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{value}</div>
                <div class="metric-lbl">{label}</div>
            </div>
        """, unsafe_allow_html=True)


def _section_row(name, desc, available: bool):
    badge_class = "badge-ready" if available else "badge-missing"
    badge_text = "✅ Will Include" if available else "⚪ Will Skip"
    st.markdown(f"""
        <div class="section-row">
            <div>
                <div class="section-name">{name}</div>
                <div class="section-desc">{desc}</div>
            </div>
            <span class="{badge_class}">{badge_text}</span>
        </div>
    """, unsafe_allow_html=True)


def _assess_available_sections():
    """
    Inspects session_state to determine which report sections have real
    data behind them. Returns a dict of section_key -> bool(available),
    plus the raw objects needed downstream so we don't re-fetch twice.
    """
    agent_report = st.session_state.get("agent_report", {})
    aq_profile = st.session_state.get("aq_profile", {})
    aq_roadmap = st.session_state.get("aq_roadmap", [])
    stats_results = st.session_state.get("stats_results", [])
    report_ai_narrative = st.session_state.get("report_ai_narrative", "")

    results = agent_report.get("results", {}) if agent_report else {}
    has_dq = "DataQualityAgent" in results
    has_insight = "InsightAgent" in results
    has_modeling = "ModelingAgent" in results

    availability = {
        "executive_summary": bool(agent_report),
        "ai_narrative": bool(report_ai_narrative),
        "data_quality": has_dq,
        "eda_highlights": has_insight,
        "statistical_findings": bool(stats_results),
        "ml_summary": has_modeling,
        "action_matrix": bool(agent_report and agent_report.get("all_recommendations")),
        "appendix": bool(aq_roadmap),
    }

    return availability, {
        "agent_report": agent_report,
        "aq_profile": aq_profile,
        "aq_roadmap": aq_roadmap,
        "stats_results": stats_results,
        "dataset_name": st.session_state.get("filename", "Uploaded Dataset"),
        "ai_narrative": report_ai_narrative
    }


def generate_ai_narrative(df: pd.DataFrame, source_name: str) -> str:
    """Uses Gemini to write a high-level executive summary of the dataset."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return f"Error configuring Gemini API: {e}"

    stats = df.describe(include='all').to_string()
    columns = ", ".join(df.columns.tolist())
    shape = f"{df.shape[0]} rows, {df.shape[1]} columns"

    prompt = f"""
    You are an expert Data Analyst and Executive Consultant.
    Write a 3-paragraph executive summary for a report based on the following dataset.
    
    Dataset Context: {source_name}
    Shape: {shape}
    Columns: {columns}
    
    Statistical Summary:
    {stats}
    
    Rules:
    - Paragraph 1: Overview of the data size and what it represents.
    - Paragraph 2: Key statistical observations (min, max, mean, top categories).
    - Paragraph 3: Potential business implications or areas for further investigation.
    - Do NOT use markdown formatting (no asterisks, no hashes). Return plain text with double line breaks between paragraphs.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating narrative: {e}"


def show():
    _inject_css()
    st.markdown('<div class="report-header">📄 Report Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="report-sub">Synthesize every completed phase into a single '
        'presentation-quality Word document.</div>',
        unsafe_allow_html=True
    )

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset found. Please upload a dataset first via **📁 Upload Data** or **SQL Connect**.")
        return

    agent_report = st.session_state.get("agent_report", {})

    if not agent_report:
        st.info(
            "ℹ️ **System Context Missing:** No Multi-Agent analysis report found. "
            "Report Generator needs this to build the Executive Summary, Data Quality, "
            "EDA, and ML sections."
        )
        st.markdown(
            "To proceed, please visit the **🤖 Multi-Agent System** page and click "
            "**Run Multi-Agent Analysis** first."
        )
        return

    # ── Stage J: AI Narrative Generator ──
    st.markdown("### AI Executive Narrative (Stage J)")
    if "report_ai_narrative" not in st.session_state:
        st.session_state["report_ai_narrative"] = ""

    if st.button("✨ Generate AI Narrative"):
        with st.spinner("Analyzing dataset statistics and generating narrative..."):
            narrative = generate_ai_narrative(st.session_state["df"], st.session_state.get("filename", "Active Dataset"))
            st.session_state["report_ai_narrative"] = narrative
            st.rerun()

    if st.session_state["report_ai_narrative"]:
        st.markdown(f'<div class="narrative-box">{st.session_state["report_ai_narrative"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    availability, context = _assess_available_sections()

    # ── Quick stats row, reusing the same engine Executive Console uses ──
    aq_profile = context["aq_profile"]
    aq_roadmap = context["aq_roadmap"]
    briefing = LocalExecutiveEngine.generate_briefing(agent_report, aq_profile, aq_roadmap)

    st.markdown("### Report Readiness")
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, f"{briefing['kpis']['health_score']}/100", "Corporate Health Index")
    _metric_card(c2, str(briefing['kpis']['high_risks']), "Critical Risks")
    _metric_card(c3, str(briefing['kpis']['action_count']), "Prescriptive Actions")
    n_ready = sum(1 for v in availability.values() if v)
    _metric_card(c4, f"{n_ready}/{len(availability)}", "Sections Ready")

    st.divider()
    st.markdown("### Report Contents Preview")
    st.caption("Sections are auto-detected from completed phases. Skipped sections require running the relevant page first.")

    _section_row(
        "1. Title & Executive Summary",
        "Narrative briefing + Corporate Health Index, generated from the Multi-Agent report.",
        availability["executive_summary"]
    )
    _section_row(
        "2. AI Data Narrative",
        "Stage J generated AI dataset narrative overview.",
        availability["ai_narrative"]
    )
    _section_row(
        "3. Data Quality Audit",
        "Missing values, duplicates, type issues, constant columns (DataQualityAgent).",
        availability["data_quality"]
    )
    _section_row(
        "4. EDA Highlights",
        "Descriptive stats, strong correlations, skewness, high-cardinality flags (InsightAgent).",
        availability["eda_highlights"]
    )
    _section_row(
        "5. Statistical Findings",
        "T-Test, Chi-Square, ANOVA, Normality results run in the Statistical Engine.",
        availability["statistical_findings"]
    )
    _section_row(
        "6. ML Model Summary",
        "Problem type, metrics (R²/Accuracy), top predictive features (ModelingAgent).",
        availability["ml_summary"]
    )
    _section_row(
        "7. Prioritized Action Matrix",
        "Ranked recommendations with priority tier, timeframe, and domain.",
        availability["action_matrix"]
    )
    _section_row(
        "8. Appendix — Analysis Roadmap",
        "Full autonomous roadmap generated in the Auto Questions phase.",
        availability["appendix"]
    )

    st.divider()
    st.markdown("### Generate Document")

    missing = [k for k, v in availability.items() if not v]
    if missing:
        st.caption(
            f"⚪ {len(missing)} section(s) will be skipped due to missing data. "
            "You can still generate a partial report."
        )

    if st.button("📄 Generate Report (.docx)", width='stretch', type="primary"):
        with st.spinner("Assembling document..."):
            docx_bytes = build_docx_report(context)
            st.session_state["report_docx_bytes"] = docx_bytes
        st.success("✅ Report assembled successfully.")

    if "report_docx_bytes" in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="⬇️ Download Report",
            data=st.session_state["report_docx_bytes"],
            file_name=f"AnalyticaOS_Report_{timestamp}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width='stretch'
        )