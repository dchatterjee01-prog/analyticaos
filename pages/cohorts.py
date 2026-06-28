"""
Stage J — Cohort Analysis Engine.
Dataset-agnostic retention analyzer. Works on the active session_state["df"].
Calculates period-over-period retention and renders a heatmap.
Includes Stage J Step 1 Dashboard pinning integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR, TEXT_COLOR, BORDER_COLOR
)

def _inject_css():
    st.markdown(f"""
    <style>
    .cohort-header {{
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
    """Utility to push the cohort heatmap to the Dashboard Composer."""
    if "dashboard_items" not in st.session_state:
        st.session_state["dashboard_items"] = []
        
    if any(item.get("title") == title for item in st.session_state["dashboard_items"]):
        st.toast(f"'{title}' is already pinned!", icon="⚠️")
        return
        
    st.session_state["dashboard_items"].append({
        "type": "plotly",
        "title": title,
        "data": fig
    })
    st.toast(f"Pinned '{title}' to Dashboard!", icon="📌")


def generate_cohort_matrix(df: pd.DataFrame, id_col: str, date_col: str, freq: str = 'M') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates the absolute count and percentage retention matrices."""
    df_cohort = df[[id_col, date_col]].dropna().copy()
    
    # Standardize datetime
    df_cohort[date_col] = pd.to_datetime(df_cohort[date_col], format="mixed")
    
    # Create period (Month or Week)
    if freq == 'M':
        df_cohort['EventPeriod'] = df_cohort[date_col].dt.to_period('M')
    else:
        df_cohort['EventPeriod'] = df_cohort[date_col].dt.to_period('W')
        
    # Find minimum period per ID (The Cohort)
    df_cohort['CohortGroup'] = df_cohort.groupby(id_col)['EventPeriod'].transform('min')
    
    # Calculate Period Offset
    if freq == 'M':
        df_cohort['PeriodIndex'] = (df_cohort['EventPeriod'].dt.year - df_cohort['CohortGroup'].dt.year) * 12 + \
                                   (df_cohort['EventPeriod'].dt.month - df_cohort['CohortGroup'].dt.month)
    else:
        # Weekly offset approximation
        df_cohort['PeriodIndex'] = (df_cohort['EventPeriod'].dt.start_time - df_cohort['CohortGroup'].dt.start_time).dt.days // 7
        
    # Pivot to absolute counts
    cohort_data = df_cohort.groupby(['CohortGroup', 'PeriodIndex'])[id_col].nunique().reset_index()
    cohort_matrix = cohort_data.pivot(index='CohortGroup', columns='PeriodIndex', values=id_col)
    
    # Convert absolute counts to percentages
    cohort_sizes = cohort_matrix.iloc[:, 0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0)
    
    # Format index for display
    cohort_matrix.index = cohort_matrix.index.astype(str)
    retention_matrix.index = retention_matrix.index.astype(str)
    
    return cohort_matrix, retention_matrix


def show():
    st.set_page_config(page_title="Cohort Analysis", page_icon="👥", layout="wide")
    _inject_css()
    
    st.title("👥 Cohort Analysis Engine")
    st.caption("Track user/entity retention over time to identify behavioral lifecycle trends.")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No active dataset. Please load data via SQL Connect or Auto Analytics first.")
        st.stop()

    df = st.session_state["df"]
    
    # Identify likely columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or 'date' in c.lower() or 'time' in c.lower()]

    st.markdown('<div class="config-card">', unsafe_allow_html=True)
    st.markdown("### Cohort Configuration")
    col1, col2, col3 = st.columns(3)
    
    id_col = col1.selectbox("Entity ID (e.g., CustomerID)", options=df.columns, index=df.columns.get_loc(num_cols[0]) if num_cols else 0)
    
    if not date_cols:
        col2.error("No date-like columns detected.")
        date_col = col2.selectbox("Timestamp Column", options=df.columns)
    else:
        date_col = col2.selectbox("Timestamp Column", options=date_cols, index=0)
        
    period = col3.selectbox("Cohort Periodicity", options=["Monthly", "Weekly"])
    freq = 'M' if period == "Monthly" else 'W'
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Generate Cohort Heatmap", type="primary"):
        with st.spinner("Processing temporal behaviors..."):
            try:
                abs_matrix, ret_matrix = generate_cohort_matrix(df, id_col, date_col, freq)
                st.session_state["cohort_abs"] = abs_matrix
                st.session_state["cohort_ret"] = ret_matrix
            except Exception as e:
                st.error(f"Error calculating cohorts: {e}")
                return

    if "cohort_ret" in st.session_state:
        st.markdown('<div class="cohort-header">Retention Heatmap</div>', unsafe_allow_html=True)
        
        ret_matrix = st.session_state["cohort_ret"]
        
        fig = px.imshow(
            ret_matrix,
            text_auto=".1%",
            aspect="auto",
            color_continuous_scale="Blues", # Maps nicely to the Scientific Blue palette
            labels=dict(x=f"Periods ({period}) since First Event", y="Cohort", color="Retention %")
        )
        fig.update_layout(
            xaxis_title=f"Periods ({period})",
            yaxis_title="Cohort Entry",
            plot_bgcolor=SURFACE_COLOR,
            paper_bgcolor=SURFACE_COLOR,
            font=dict(color=TEXT_COLOR)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_pin, col_empty = st.columns([1, 4])
        with col_pin:
            st.button("📌 Pin to Dashboard", key="pin_cohort", on_click=_pin_to_dashboard, args=(f"{period} Retention Cohorts", fig))

        with st.expander("📊 View Absolute Counts Matrix"):
            st.dataframe(st.session_state["cohort_abs"].astype(str), use_container_width=True)