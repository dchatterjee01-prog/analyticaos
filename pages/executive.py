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

        high_severity_count = sum(1 for f in all_findings if f.get("severity") == "high")
        med_severity_count = sum(1 for f in all_findings if f.get("severity") == "medium")

        base_score = 100
        deductions = (high_severity_count * 15) + (med_severity_count * 5)
        health_score = max(10, min(100, base_score - deductions))

        row_count = aq_profile.get("n_rows", 0) if aq_profile else 0
        col_count = aq_profile.get("n_cols", 0) if aq_profile else 0

        status_terms = {
            "ok": "robust with strong technical foundations",
            "warning": "viable but requires target structural optimizations",
            "error": "exposed to critical data integrity and modeling risks"
        }
        current_term = status_terms.get(overall_status, "stable")

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

        def _shares_keyword(rec_text: str, finding_text: str) -> bool:
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


class PrescriptiveEngine:
    _CANDIDATE_FI_KEYS = ["feature_importance", "feature_importances", "importances"]

    @staticmethod
    def _extract_feature_importance(agent_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not agent_report:
            return []

        raw = None
        for key in PrescriptiveEngine._CANDIDATE_FI_KEYS:
            if key in agent_report and agent_report[key]:
                raw = agent_report[key]
                break

        if raw is None:
            return []

        normalized = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    feat = item.get("feature") or item.get("name")
                    imp = item.get("importance") or item.get("value")
                    if feat is not None and imp is not None:
                        normalized.append({"feature": str(feat), "importance": float(imp)})
        elif isinstance(raw, dict):
            for feat, imp in raw.items():
                try:
                    normalized.append({"feature": str(feat), "importance": float(imp)})
                except (TypeError, ValueError):
                    continue

        normalized.sort(key=lambda x: x["importance"], reverse=True)
        return normalized

    @staticmethod
    def _extract_optimization_levers(metric_direction: str) -> List[Dict[str, Any]]:
        levers = []

        opt_channels = st.session_state.get("opt_channels")
        if opt_channels is not None and not opt_channels.empty:
            try:
                top_channel = opt_channels.sort_values(
                    "Return per $ (ROI)", ascending=False
                ).iloc[0]
                levers.append({
                    "lever": f"Reallocate budget toward '{top_channel['Channel']}'",
                    "source": "Optimization Engine — Budget Allocation",
                    "rationale": f"Highest return per dollar in current channel set ({top_channel['Return per $ (ROI)']:.2f}x).",
                    "impact_estimate": f"+{top_channel['Return per $ (ROI)']:.1f}x return per $ shifted"
                })
            except (KeyError, IndexError):
                pass

        opt_tasks = st.session_state.get("opt_tasks")
        if opt_tasks is not None and not opt_tasks.empty:
            try:
                top_task = opt_tasks.sort_values(
                    "Output per Unit", ascending=False
                ).iloc[0]
                levers.append({
                    "lever": f"Shift resource units toward '{top_task['Task']}'",
                    "source": "Optimization Engine — Resource Allocation",
                    "rationale": f"Highest output per unit in current task set ({top_task['Output per Unit']:.1f}).",
                    "impact_estimate": f"+{top_task['Output per Unit']:.1f} output per unit shifted"
                })
            except (KeyError, IndexError):
                pass

        return levers

    @staticmethod
    def generate_recommendations(
        agent_report: Dict[str, Any],
        metric_name: str,
        direction: str
    ) -> Dict[str, Any]:
        all_findings = agent_report.get("all_findings", []) if agent_report else []
        fi_data = PrescriptiveEngine._extract_feature_importance(agent_report)
        opt_levers = PrescriptiveEngine._extract_optimization_levers(direction)

        levers = []

        for fi in fi_data[:3]:
            verb = "Increase focus on" if direction == "increase" else "Reduce dependency on"
            levers.append({
                "lever": f"{verb} '{fi['feature']}'",
                "source": "ModelingAgent — Feature Importance",
                "rationale": f"Ranked driver of model predictions (relative importance: {fi['importance']:.3f}).",
                "impact_estimate": "High — top-ranked model feature"
            })

        levers.extend(opt_levers)

        if len(levers) < 3:
            severity_rank = {"high": 2, "medium": 1, "low": 0}
            sortable_findings = sorted(
                [f for f in all_findings if f.get("severity") in ("high", "medium")],
                key=lambda f: severity_rank.get(f.get("severity"), 0),
                reverse=True
            )
            for f in sortable_findings:
                if len(levers) >= 3:
                    break
                levers.append({
                    "lever": f.get("title", "Address flagged finding"),
                    "source": "InsightAgent — Diagnostic Finding",
                    "rationale": f.get("detail", "No further detail available."),
                    "impact_estimate": "Medium — resolving this finding likely stabilizes the metric"
                })

        top_levers = levers[:3]
        for idx, lever in enumerate(top_levers):
            lever["rank"] = idx + 1

        return {
            "metric_name": metric_name,
            "direction": direction,
            "levers": top_levers,
            "fi_available": len(fi_data) > 0,
            "opt_available": len(opt_levers) > 0
        }


class ConsultingFrameworksEngine:
    """
    Stage D, Step Group 3 — Executive Consulting Frameworks.
    Template-based synthesis from existing agent_report/aq_profile/briefing
    data. No live LLM call. Scope this step: SWOT + Scenario/Risk Analysis.
    """

    @staticmethod
    def generate_swot(
        agent_report: Dict[str, Any],
        aq_profile: Dict[str, Any],
        briefing: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        all_findings = agent_report.get("all_findings", []) if agent_report else []
        all_recs = agent_report.get("all_recommendations", []) if agent_report else []
        health_score = briefing.get("health_score", 0)
        row_count = aq_profile.get("n_rows", 0) if aq_profile else 0
        col_count = aq_profile.get("n_cols", 0) if aq_profile else 0

        low_findings = [f for f in all_findings if f.get("severity") == "low"]
        med_findings = [f for f in all_findings if f.get("severity") == "medium"]
        high_findings = [f for f in all_findings if f.get("severity") == "high"]

        strengths = []
        if health_score >= 75:
            strengths.append(f"Strong overall data health (Corporate Health Index: {health_score}/100).")
        if row_count > 0 and col_count > 0:
            strengths.append(f"Substantial analytical base: {row_count:,} observations across {col_count} dimensions.")
        if low_findings:
            strengths.append(f"{len(low_findings)} dimensions flagged at low risk, indicating stable structural areas.")
        if not strengths:
            strengths.append("Dataset is under active diagnostic review; no confirmed strengths yet.")

        weaknesses = []
        if high_findings:
            weaknesses.append(f"{len(high_findings)} high-severity findings requiring structural remediation.")
        if med_findings:
            weaknesses.append(f"{len(med_findings)} medium-severity findings indicating tactical gaps.")
        if health_score < 75:
            weaknesses.append(f"Corporate Health Index below target threshold ({health_score}/100).")
        if not weaknesses:
            weaknesses.append("No material weaknesses identified in current diagnostic pass.")

        opportunities = []
        if all_recs:
            opportunities.append(f"{len(all_recs)} prescriptive actions identified, each representing a near-term improvement opportunity.")
        if health_score >= 75:
            opportunities.append("Current data readiness supports expansion into predictive modeling and forecasting use cases.")
        if not opportunities:
            opportunities.append("Opportunities will become clearer once additional analysis modules are run.")

        threats = []
        if high_findings:
            threats.append("Unresolved high-severity findings risk downstream model and decision validity if left unaddressed.")
        if health_score < 40:
            threats.append("Health Index in critical range — continued use of current dataset for decisions carries material risk.")
        if not threats:
            threats.append("No acute threats identified; monitor for drift as new data is ingested.")

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "opportunities": opportunities,
            "threats": threats
        }

    @staticmethod
    def generate_risk_analysis(
        agent_report: Dict[str, Any],
        briefing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        all_findings = agent_report.get("all_findings", []) if agent_report else []
        health_score = briefing.get("health_score", 0)

        severity_to_risk = {"high": "Critical", "medium": "Moderate", "low": "Minor"}
        likelihood_map = {"high": "High", "medium": "Medium", "low": "Low"}

        risks = []
        for f in all_findings:
            sev = f.get("severity", "low")
            risks.append({
                "risk": f.get("title", "Unclassified finding"),
                "severity": severity_to_risk.get(sev, "Minor"),
                "likelihood": likelihood_map.get(sev, "Low"),
                "exposure": f.get("detail", "No further detail available."),
                "mitigation": "Immediate remediation" if sev == "high" else (
                    "Scheduled review" if sev == "medium" else "Monitor"
                )
            })

        if health_score < 40:
            risks.insert(0, {
                "risk": "Overall data asset health in critical range",
                "severity": "Critical",
                "likelihood": "High",
                "exposure": f"Corporate Health Index at {health_score}/100 — decisions based on this dataset carry elevated risk.",
                "mitigation": "Immediate remediation"
            })

        severity_rank = {"Critical": 2, "Moderate": 1, "Minor": 0}
        risks.sort(key=lambda r: severity_rank.get(r["severity"], 0), reverse=True)
        return risks


def _inject_css():
    st.markdown(f"""
        <style>
        .exec-header {{ color: {PRIMARY_COLOR}; font-size: 28px; font-weight: 700; margin-bottom: 4px; }}
        .exec-sub {{ color: {MUTED_COLOR}; font-size: 14px; margin-bottom: 24px; }}
        .metric-card {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; padding: 16px; border-radius: 6px; text-align: center; }}
        .metric-val {{ color: {TEXT_COLOR}; font-size: 24px; font-weight: 700; }}
        .metric-lbl {{ color: {MUTED_COLOR}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
        .narrative-box {{ background-color: {BACKGROUND_COLOR}; border-left: 4px solid {PRIMARY_COLOR}; padding: 16px; border-radius: 0 6px 6px 0; font-size: 15px; line-height: 1.6; color: {TEXT_COLOR}; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .lever-card {{ background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-left: 4px solid {PRIMARY_COLOR}; padding: 14px 16px; border-radius: 0 6px 6px 0; margin-bottom: 12px; }}
        .lever-title {{ color: {TEXT_COLOR}; font-size: 16px; font-weight: 700; }}
        .lever-source {{ color: {MUTED_COLOR}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; margin-top: 2px; }}
        .lever-rationale {{ color: {TEXT_COLOR}; font-size: 14px; margin-top: 8px; line-height: 1.5; }}
        .lever-impact {{ color: {PRIMARY_COLOR}; font-size: 13px; font-weight: 600; margin-top: 6px; }}
        .swot-card {{ border-radius: 8px; padding: 16px; height: 100%; }}
        .swot-strengths {{ background-color: {SUCCESS_BG}; border: 1px solid {SUCCESS_COLOR}; }}
        .swot-weaknesses {{ background-color: {ERROR_BG}; border: 1px solid {ERROR_COLOR}; }}
        .swot-opportunities {{ background-color: {WARNING_BG}; border: 1px solid {WARNING_COLOR}; }}
        .swot-threats {{ background-color: {BACKGROUND_COLOR}; border: 1px solid {BORDER_COLOR}; }}
        .swot-title {{ font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }}
        .swot-item {{ font-size: 13.5px; color: {TEXT_COLOR}; margin-bottom: 6px; line-height: 1.4; }}
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

def _render_gauge(score):
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

def _render_briefing_tab(agent_report, aq_profile, aq_roadmap):
    briefing = LocalExecutiveEngine.generate_briefing(agent_report, aq_profile, aq_roadmap)

    if briefing["status"] == "pending":
        st.info(f"ℹ️ **System Context Missing:** {briefing['narrative_summary']}")
        st.markdown("To generate this briefing, please visit the **🤖 Multi-Agent System** page and click **Run Multi-Agent Analysis** first.")
        return briefing

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

    return briefing


def _render_prescriptive_tab(agent_report):
    st.markdown("### What Should We Do?")
    st.markdown(
        "Choose a business metric and direction. AnalyticaOS will surface the "
        "**top 3 levers** to move it, synthesized from your ML model's feature "
        "importance, this session's Optimization Engine results, and flagged "
        "diagnostic findings."
    )

    col_metric, col_dir = st.columns([2, 1])
    with col_metric:
        metric_name = st.text_input(
            "Business metric",
            value="Revenue",
            key="prescriptive_metric_input"
        )
    with col_dir:
        direction = st.selectbox(
            "Direction",
            options=["increase", "decrease"],
            key="prescriptive_direction_input"
        )

    generate_clicked = st.button(
        "Generate Recommendations", type="primary", key="prescriptive_generate_btn"
    )

    if not generate_clicked:
        return

    if not metric_name.strip():
        st.error("Enter a metric name before generating recommendations.")
        return

    result = PrescriptiveEngine.generate_recommendations(agent_report, metric_name.strip(), direction)

    if not result["fi_available"]:
        st.caption(
            "ℹ️ No ML feature importance data found in this session's agent report — "
            "levers below are drawn from Optimization results and diagnostic findings only."
        )
    if not result["opt_available"]:
        st.caption(
            "ℹ️ No Optimization Engine results found in this session — "
            "visit **🧮 Optimization Engine** and run a solve for allocation-based levers."
        )

    st.markdown(f"#### Top Levers to **{result['direction'].title()}** *{result['metric_name']}*")

    if not result["levers"]:
        st.warning(
            "No levers could be synthesized — no feature importance, optimization "
            "results, or flagged findings are available yet this session."
        )
        return

    for lever in result["levers"]:
        st.markdown(f"""
            <div class="lever-card">
                <div class="lever-title">#{lever['rank']} — {lever['lever']}</div>
                <div class="lever-source">{lever['source']}</div>
                <div class="lever-rationale">{lever['rationale']}</div>
                <div class="lever-impact">Estimated Impact: {lever['impact_estimate']}</div>
            </div>
        """, unsafe_allow_html=True)


def _render_swot_quadrant(col, title, items, css_class):
    with col:
        items_html = "".join(f'<div class="swot-item">• {item}</div>' for item in items)
        st.markdown(f"""
            <div class="swot-card {css_class}">
                <div class="swot-title">{title}</div>
                {items_html}
            </div>
        """, unsafe_allow_html=True)


def _render_frameworks_tab(agent_report, aq_profile, briefing):
    st.markdown("### SWOT Analysis")

    if briefing.get("status") == "pending":
        st.info("ℹ️ Run **🤖 Multi-Agent System** analysis first to generate SWOT and Risk Analysis.")
        return

    swot = ConsultingFrameworksEngine.generate_swot(agent_report, aq_profile, briefing)

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    _render_swot_quadrant(row1_col1, "Strengths", swot["strengths"], "swot-strengths")
    _render_swot_quadrant(row1_col2, "Weaknesses", swot["weaknesses"], "swot-weaknesses")
    _render_swot_quadrant(row2_col1, "Opportunities", swot["opportunities"], "swot-opportunities")
    _render_swot_quadrant(row2_col2, "Threats", swot["threats"], "swot-threats")

    st.divider()
    st.markdown("### Scenario / Risk Analysis")

    risks = ConsultingFrameworksEngine.generate_risk_analysis(agent_report, briefing)

    if not risks:
        st.success("🎉 No risks identified in current diagnostic pass.")
        return

    df_risks = pd.DataFrame(risks)

    def _style_severity(val):
        if val == "Critical": return f"background-color: {ERROR_BG}; color: {ERROR_COLOR}; font-weight: bold;"
        elif val == "Moderate": return f"background-color: {WARNING_BG}; color: {WARNING_COLOR}; font-weight: bold;"
        return f"background-color: {SUCCESS_BG}; color: {SUCCESS_COLOR};"

    styled_risks = df_risks.style.map(_style_severity, subset=["severity"])
    st.dataframe(
        styled_risks,
        column_config={
            "risk": "Identified Risk",
            "severity": "Severity",
            "likelihood": "Likelihood",
            "exposure": "Exposure Detail",
            "mitigation": "Recommended Mitigation"
        },
        hide_index=True,
        width='stretch'
    )


def show():
    _inject_css()
    st.markdown('<div class="exec-header">🏛️ Executive Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="exec-sub">Strategic operational briefing and prioritized core actions.</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset found. Please upload a dataset in the Data Hub first.")
        return

    agent_report = st.session_state.get("agent_report", {})
    aq_profile = st.session_state.get("aq_profile", {})
    aq_roadmap = st.session_state.get("aq_roadmap", [])

    tab_briefing, tab_prescriptive, tab_frameworks = st.tabs(
        ["📋 Strategic Briefing", "🎯 Prescriptive Analytics", "🧭 Consulting Frameworks"]
    )

    with tab_briefing:
        briefing = _render_briefing_tab(agent_report, aq_profile, aq_roadmap)

    with tab_prescriptive:
        _render_prescriptive_tab(agent_report)

    with tab_frameworks:
        _render_frameworks_tab(agent_report, aq_profile, briefing)