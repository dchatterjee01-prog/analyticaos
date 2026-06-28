"""
Stage J — Funnel Analysis Engine.
Dataset-agnostic funnel analyzer. Works on session_state["df"].
Calculates step-by-step conversion and abandonment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR, TEXT_COLOR, BORDER_COLOR
)

def _inject_css():
    st.markdown(f"""
    <style>
    .funnel-header {{
        color: {PRIMARY_COLOR};
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 0.6rem;
        margin: 1.2rem 0 1rem 0;
    }}
    .config-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True)

def _pin_to_dashboard(title: str, fig):
    if "dashboard_items" not in st.session_state:
        st.session_state["dashboard_items"] = []
    
    st.session_state["dashboard_items"].append({
        "type": "plotly",
        "title": title,
        "data": fig
    })
    st.toast(f"Pinned '{title}' to Dashboard!", icon="📌")

def calculate_funnel(df: pd.DataFrame, id_col: str, event_col: str, steps: list) -> pd.DataFrame:
    """Computes conversion counts for a sequence of events."""
    data = []
    total_users = df[id_col].nunique()
    
    for step in steps:
        # Get users who performed this event
        users = df[df[event_col] == step][id_col].unique()
        data.append({"Step": step, "Count": len(users)})
        
    return pd.DataFrame(data)

def show():
    st.set_page_config(page_title="Funnel Analysis", page_icon="🎯", layout="wide")
    _inject_css()
    
    st.title("🎯 Funnel Analysis Engine")
    st.caption("Measure conversion drop-off across defined event sequences.")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset. Please load data via SQL Connect or Auto Analytics first.")
        st.stop()

    df = st.session_state["df"]
    
    st.markdown('<div class="config-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    id_col = col1.selectbox("User/Entity ID Column", options=df.columns)
    event_col = col2.selectbox("Event Name Column", options=df.columns)
    
    # Allow user to select steps from unique values in the event column
    unique_events = df[event_col].unique().tolist()
    steps = st.multiselect("Define Funnel Steps (in order)", options=unique_events)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Analyze Funnel", type="primary") and steps:
        with st.spinner("Calculating conversion..."):
            funnel_df = calculate_funnel(df, id_col, event_col, steps)
            
            fig = px.funnel(
                funnel_df, x='Count', y='Step',
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig.update_layout(plot_bgcolor=SURFACE_COLOR, paper_bgcolor=SURFACE_COLOR)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.button("📌 Pin to Dashboard", on_click=_pin_to_dashboard, args=("Conversion Funnel", fig))
            st.dataframe(funnel_df.astype(str), use_container_width=True)
    elif not steps:
        st.info("Select at least two steps to visualize a funnel.")