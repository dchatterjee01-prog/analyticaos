# pages/executive.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR, TEXT_COLOR,
    BORDER_COLOR, MUTED_COLOR, SUCCESS_COLOR, SUCCESS_BG,
    WARNING_COLOR, WARNING_BG, ERROR_COLOR, ERROR_BG
)

class LocalExecutiveEngine:
    """
    Embedded, self-contained business logic to synthesize data postures,
    dataset profiles, and roadmaps into strategic executive briefings.
    """
    @staticmethod
    def generate_briefing(agent_report: Dict[str, Any], aq_profile: Dict[str, Any], aq_roadmap: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not agent_report:
            return {
                "status": "pending",
                "health_score": 0,
                "narrative_summary": "No Multi-Agent analysis report found. Please execute the Multi-Agent System analysis first.",
                "action_matrix": [],
                "kpis": {"health_score": 0, "high_risks": 0, "med_risks": 0, "total_insights": 0, "action_count": 0}
            }

        all_findings = agent_report.get("all_findings", [])
        all_recs = agent_report.get("all_recommendations", [])
        overall_status = agent_report.get("overall_status", "ok").lower()

        # 1. Calculate Strategic Health Score
        # FIX (Phase 11 prerequisite): finding dicts use key "severity" with
        # values "low"/"medium"/"high" — NOT "status" with "error"/"warning".
        # The old check (f.get("status") == "error") never matched anything,
        # since findings never carry a "status" key (only AgentResult does),
        # so health_score was always pinned at 100 regardless of real findings.
        high_severity_count = sum(1 for f in all_findings if f.get("severity") == "high")
        med_severity_count = sum(1 for f in all_findings if f.get("severity") == "medium")

        base_score = 100
        deductions = (high_severity_count * 15) + (med_severity_count * 5)
        health_score = max(10, min(100, base_score - deductions))

        # 2. Extract profile dimensions
        # FIX (Phase 11 prerequisite): aq_profile (set in pages/questions.py)
        # actually uses keys "n_rows" / "n_cols" — NOT "total_rows" / "total_cols".
        # The old keys never existed in aq_profile, so row_count/col_count
        # were always 0 in the narrative.
        row_count = aq_profile.get("n_rows", 0) if aq_profile else 0
        col_count = aq_profile.get("n_cols", 0) if aq_profile else 0

        status_terms = {
            "ok": "robust with strong technical foundations",
            "warning": "viable but requires target structural optimizations",
            "error": "exposed to critical data integrity and modeling risks"
        }
        current_term = status_terms.get(overall_status, "stable")

        # 3. Synthesize Strategic Briefing Narrative
        narrative = (
            f"AnalyticaOS has completed a comprehensive operational audit of the source dataset containing {row_count:,} observations "
            f"across {col_count} distinct structural dimensions. The global asset posture is currently evaluated as **{current_term.upper()}** "
            f"(Corporate Health Index: **{health_score}/100**). "
        )

        if high_severity_count > 0:
            narrative += f"Immediate executive intervention is mandatory to address {high_severity_count} critical structural bottlenecks threatening analytical validity. "
        elif med_severity_count > 0:
            narrative += f"Management should prioritize {med_severity_count} operational optimizations to prevent downstream statistical drift. "
        else:
            narrative += "The dataset demonstrates exceptional structural readiness for immediate predictive modeling and business intelligence deployment. "

        if len(all_recs) > 0:
            narrative += f"A structured roadmap containing {len(all_recs)} prescriptive actions has been compiled to de-risk asset monetization."

        # 4. Build Prioritized Action Matrix
        # FIX: the previous version tried to match f.get("agent_name") as a
        # substring inside the recommendation text (e.g. checking whether
        # "dataqualityagent" appears in "Impute or drop 'sales' — 10.0%
        # missing."). Finding dicts never carry an "agent_name" key (only
        # title/detail/severity — see agents/base_agent.py AgentResult),
        # and even if they did, agent names never literally appear inside
        # plain-language recommendation sentences. That check was always
        # False, so priority_tier fell through to a weak "optimize"/"fix"
        # keyword search, and almost everything — including recommendations
        # generated from HIGH severity findings like duplicate rows or low
        # model accuracy — was mislabeled "Low" priority.
        #
        # Real fix: findings and recommendations frequently share concrete
        # nouns (column names in quotes, key terms like "duplicate",
        # "accuracy", "correlat-"). We match each recommendation against
        # every finding's title+detail using simple word overlap, and take
        # the highest severity among findings that share at least one
        # meaningful word (length > 3, to skip "the"/"and"/etc).
        def _shares_keyword(rec_text: str, finding_text: str) -> bool:
            # Require 2+ shared meaningful words, not just one — a single
            # shared column name (e.g. 'revenue') can appear in two
            # completely unrelated findings (a missing-value finding and an
            # unrelated ML finding both mentioning the same column), which
            # caused false-positive severity matches in testing.
            rec_words = set(w.strip("'\".,()%-") for w in rec_text.lower().split() if len(w) > 3)
            finding_words = set(w.strip("'\".,()%-") for w in finding_text.lower().split() if len(w) > 3)
            return len(rec_words & finding_words) >= 2

        severity_rank = {"high": 2, "medium": 1, "low": 0}

        prioritized_actions = []
        for idx, rec in enumerate(all_recs):
            if not rec:
                continue

            matched_severities = [
                f.get("severity", "low")
                for f in all_findings
                if _shares_keyword(rec, f.get("title", "") + " " + f.get("detail", ""))
            ]

            if matched_severities:
                best_severity = max(matched_severities, key=lambda s: severity_rank.get(s, 0))
            else:
                # No matching finding found — fall back to keyword heuristic
                # on the recommendation text itself rather than silently
                # defaulting to Low.
                best_severity = "medium" if any(
                    kw in rec.lower() for kw in ["duplicate", "accuracy", "low r", "missing", "critical"]
                ) else "low"

            priority_tier = {"high": "High", "medium": "Medium", "low": "Low"}.get(best_severity, "Low")
            timeframe = (
                "Immediate (0-48h)" if priority_tier == "High"
                else "Tactical (1-2 weeks)" if priority_tier == "Medium"
                else "Strategic (Quarterly)"
            )

            prioritized_actions.append({
                "id": f"STRAT-{idx+1:02d}",
                "action": rec,
                "priority": priority_tier,
                "timeframe": timeframe,
                "domain": "Data Governance" if "quality" in rec.lower() or "missing" in rec.lower() else "Advanced Analytics"
            })

        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        prioritized_actions.sort(key=lambda x: priority_order.get(x["priority"], 3))

        kpis = {
            "health_score": health_score,
            "high_risks": high_severity_count,
            "med_risks": med_severity_count,
            "total_insights": len([f for f in all_findings if f.get("severity") == "low"]),
            "action_count": len(prioritized_actions)
        }

        return {
            "status": "success" if overall_status != "error" else "error",
            "health_score": health_score,
            "narrative_summary": narrative,
            "action_matrix": prioritized_actions,
            "kpis": kpis
        }

def _inject_css():
    """Injects layout styling elements bound completely to the core color palette."""
    st.markdown(f"""
        <style>
        .exec-header {{ color: {PRIMARY_COLOR}; font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
        .exec-sub {{ color: {MUTED_COLOR}; font-size: 14px; margin-bottom: 24px; }}
        .metric-card {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; padding: 16px; border-radius: 6px; text-align: center; }}
        .metric-val {{ color: {TEXT_COLOR}; font-size: 24px; font-weight: 700; }}
        .metric-lbl {{ color: {MUTED_COLOR}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
        .narrative-box {{ background-color: {BACKGROUND_COLOR}; border-left: 4px solid {PRIMARY_COLOR}; padding: 16px; border-radius: 0 6px 6px 0; font-size: 15px; line-height: 1.6; color: {TEXT_COLOR}; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        </style>
    """, unsafe_allow_html=True)

def _metric_card(col, value, label):
    """Renders custom styled executive visual panels."""
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{value}</div>
                <div class="metric-lbl">{label}</div>
            </div>
        """, unsafe_allow_html=True)

def _render_gauge(score):
    """Generates an enterprise-grade vector dial layout."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_COLOR},
            'bar': {'color': PRIMARY_COLOR},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': BORDER_COLOR,
            'steps': [
                {'range': [0, 40], 'color': ERROR_BG},
                {'range': [40, 75], 'color': WARNING_BG},
                {'range': [75, 100], 'color': SUCCESS_BG}
            ]
        }
    ))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': TEXT_COLOR, 'family': "Arial"}
    )
    return fig

def show():
    """Main routing dashboard execution framework."""
    _inject_css()
    st.markdown('<div class="exec-header">🏛️ Executive Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="exec-sub">Strategic operational briefing and prioritized core actions.</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset found. Please upload a dataset in the Data Hub first.")
        return

    agent_report = st.session_state.get("agent_report", {})
    aq_profile = st.session_state.get("aq_profile", {})
    aq_roadmap = st.session_state.get("aq_roadmap", [])

    briefing = LocalExecutiveEngine.generate_briefing(agent_report, aq_profile, aq_roadmap)

    if briefing["status"] == "pending":
        st.info(f"ℹ️ **System Context Missing:** {briefing['narrative_summary']}")
        st.markdown("To generate this briefing, please visit the **🤖 Multi-Agent System** page and click **Run Multi-Agent Analysis** first.")
        return

    # Split Pane Metric Layout
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("### Strategic Briefing Narrative")
        st.markdown(f'<div class="narrative-box">{briefing["narrative_summary"]}</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown("### Corporate Health")
        st.plotly_chart(_render_gauge(briefing["health_score"]), width='stretch')

    st.divider()
    st.markdown("### Core Engagement Metrics")
    kpis = briefing["kpis"]
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, f"{kpis['health_score']}/100", "Corporate Health Index")
    _metric_card(c2, str(kpis['high_risks']), "Critical Risks Flagged")
    _metric_card(c3, str(kpis['med_risks']), "Tactical Risks Flagged")
    _metric_card(c4, str(kpis['action_count']), "Prescriptive Priorities")

    st.divider()
    st.markdown("### Prioritized Action Matrix")
    matrix_data = briefing["action_matrix"]

    if not matrix_data:
        st.success("🎉 No remediation actions required. Operational integrity posture is optimal.")
    else:
        df_matrix = pd.DataFrame(matrix_data)
        def _style_priority(val):
            if val == "High": return f"background-color: {ERROR_BG}; color: {ERROR_COLOR}; font-weight: bold;"
            elif val == "Medium": return f"background-color: {WARNING_BG}; color: {WARNING_COLOR}; font-weight: bold;"
            return f"background-color: {SUCCESS_BG}; color: {SUCCESS_COLOR};"

        styled_df = df_matrix.style.map(_style_priority, subset=["priority"])
        st.dataframe(
            styled_df,
            column_config={
                "id": "Strategic ID",
                "action": "Prescriptive Action Item",
                "priority": "Priority Tier",
                "timeframe": "Remediation Window",
                "domain": "Operational Domain"
            },
            hide_index=True,
            width='stretch'
        )