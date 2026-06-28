"""
Stage J — Static Dashboard Composer.
Allows users to view and arrange pinned charts, metrics, and dataframes
from across the AnalyticaOS session.
Uses st.session_state["dashboard_items"] for ephemeral storage.
"""

import streamlit as st
import pandas as pd
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR, TEXT_COLOR, BORDER_COLOR
)

def _inject_css():
    st.markdown(f"""
    <style>
    .dashboard-header {{
        color: {PRIMARY_COLOR};
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 0.6rem;
        margin: 1.2rem 0 1rem 0;
    }}
    .dashboard-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    </style>
    """, unsafe_allow_html=True)


def _render_item(item: dict, index: int):
    """Renders a single pinned item based on its type."""
    st.markdown(f"**{item.get('title', 'Untitled')}**")
    
    item_type = item.get("type")
    
    if item_type == "metric":
        col1, col2 = st.columns(2)
        col1.metric(label=item.get("label", ""), value=item.get("value", ""))
    
    elif item_type == "dataframe":
        df = item.get("data")
        if isinstance(df, pd.DataFrame):
            st.dataframe(df.astype(str), use_container_width=True)
        else:
            st.error("Invalid dataframe format.")
            
    elif item_type == "plotly":
        fig = item.get("data")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Invalid chart object.")
            
    if st.button("🗑️ Remove", key=f"dash_remove_{index}"):
        st.session_state["dashboard_items"].pop(index)
        st.rerun()


def show():
    st.set_page_config(page_title="Dashboard Composer", page_icon="📊", layout="wide")
    _inject_css()
    
    st.title("📊 Dashboard Composer")
    st.caption("Your personalized view of pinned metrics, charts, and tables from this session.")
    
    # Initialize dashboard state if it doesn't exist
    if "dashboard_items" not in st.session_state:
        st.session_state["dashboard_items"] = []
        
    items = st.session_state["dashboard_items"]
    
    col_head, col_action = st.columns([4, 1])
    with col_head:
        st.markdown('<div class="dashboard-header">Pinned Insights</div>', unsafe_allow_html=True)
    with col_action:
        if st.button("🗑️ Clear Dashboard", use_container_width=True) and items:
            st.session_state["dashboard_items"] = []
            st.rerun()

    st.divider()

    if not items:
        st.info("Your dashboard is empty. Navigate to the Insights or Ask Your Data pages to pin charts and metrics here.")
        return

    # Dynamic Grid Layout (2 columns)
    cols = st.columns(2)
    for idx, item in enumerate(items):
        col = cols[idx % 2]
        with col:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            _render_item(item, idx)
            st.markdown('</div>', unsafe_allow_html=True)