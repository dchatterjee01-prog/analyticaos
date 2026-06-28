"""
Stage I & J — Autonomous Analytics SQL Agent UI (Updated with Pinning).
Runs auto-generated analytical queries, displays results, feeds
the best result into the pipeline, and allows pinning to the Dashboard.
"""

import streamlit as st
import pandas as pd
from connections.db_engine import get_engine, get_conn_meta
from connections.auto_sql_agent import (
    run_auto_analysis, best_result_for_pipeline,
    AutoSQLResult, AutoQuery,
    QUERY_CATEGORIES, CATEGORY_LABELS, CATEGORY_ICONS,
)
from connections.bridge import apply_bridge
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR, TEXT_COLOR,
    BORDER_COLOR, SUCCESS_BG, SUCCESS_COLOR,
    WARNING_BG, WARNING_COLOR, ERROR_BG, ERROR_COLOR,
)

def _inject_css():
    st.markdown(f"""
    <style>
    .section-header {{
        color: {PRIMARY_COLOR};
        font-size: 1.05rem;
        font-weight: 700;
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.5rem 0;
    }}
    .query-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True)

def _pin_to_dashboard(title: str, df: pd.DataFrame):
    """Utility to push a dataframe to the Dashboard Composer state."""
    if "dashboard_items" not in st.session_state:
        st.session_state["dashboard_items"] = []
        
    # Check for duplicates to prevent spamming the dashboard
    if any(item.get("title") == title for item in st.session_state["dashboard_items"]):
        st.toast(f"'{title}' is already on the dashboard!", icon="⚠️")
        return
        
    st.session_state["dashboard_items"].append({
        "type": "dataframe",
        "title": title,
        "data": df.copy()
    })
    st.toast(f"Pinned '{title}' to Dashboard!", icon="📌")

def _render_query_card(q: AutoQuery, idx: int):
    st.markdown('<div class="query-card">', unsafe_allow_html=True)
    st.markdown(f"**{q.title}**")
    st.caption(q.rationale)
    
    if q.error:
        st.error(f"Execution failed: {q.error}")
        st.code(q.sql, language="sql")
    elif q.result_df is not None:
        st.dataframe(q.result_df.astype(str), use_container_width=True)
        
        col1, col2, col3 = st.columns([2, 2, 4])
        with col1:
            st.button("📥 Load as Active Dataset", key=f"auto_load_{idx}", 
                      on_click=lambda q=q: st.session_state.update({
                          "df": apply_bridge(q.result_df)[0], 
                          "data_source": f"Auto SQL: {q.title}"
                      }))
        with col2:
            # Stage J Integration: The Pin Button
            st.button("📌 Pin to Dashboard", key=f"auto_pin_{idx}", 
                      on_click=_pin_to_dashboard, args=(q.title, q.result_df))
    st.markdown('</div>', unsafe_allow_html=True)

def show():
    st.set_page_config(page_title="Auto SQL Analytics", page_icon="🤖", layout="wide")
    _inject_css()
    
    st.title("🤖 Autonomous Analytics Agent")
    st.caption("Gemini explores your schema and generates key business insights automatically.")

    if not get_engine():
        st.warning("⚠️ No active database connection. Connect via **SQL Connect**.")
        st.stop()

    meta = get_conn_meta()
    
    if st.button("🚀 Run Autonomous Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {meta['database']}..."):
            try:
                result = run_auto_analysis()
                st.session_state["auto_sql_last_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
                
    if "auto_sql_last_result" in st.session_state:
        result: AutoSQLResult = st.session_state["auto_sql_last_result"]
        
        st.markdown('<div class="section-header">Generated Insights</div>', unsafe_allow_html=True)
        
        if not result.queries:
            st.info("No queries were generated.")
        else:
            for cat in QUERY_CATEGORIES:
                cat_queries = [q for q in result.queries if q.category == cat]
                if cat_queries:
                    st.markdown(f"### {CATEGORY_ICONS.get(cat, '')} {CATEGORY_LABELS.get(cat, cat)}")
                    for i, q in enumerate(cat_queries):
                        global_idx = result.queries.index(q)
                        _render_query_card(q, global_idx)