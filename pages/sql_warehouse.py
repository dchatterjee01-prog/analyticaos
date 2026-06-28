"""
Stage I — Data Warehouse Schema Advisor UI.
Analyzes connected schema via Gemini and proposes a star schema redesign.
DDL is display/download only — never auto-executed against the database.
"""

import streamlit as st
import pandas as pd
from connections.db_engine import get_engine, get_conn_meta
from connections.warehouse_advisor import (
    analyze_schema_for_warehouse, ddl_safe_for_display, WarehouseAdvice,
)
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
    .table-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }}
    .fact-badge {{
        background: {WARNING_BG};
        color: {WARNING_COLOR};
        border: 1px solid {WARNING_COLOR};
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.5rem;
    }}
    .dim-badge {{
        background: {SUCCESS_BG};
        color: {SUCCESS_COLOR};
        border: 1px solid {SUCCESS_COLOR};
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 0.5rem;
    }}
    .metric-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .metric-val {{
        font-size: 1.8rem;
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
    .rationale-box {{
        background: {SURFACE_COLOR};
        border-left: 4px solid {ACCENT_COLOR};
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.1rem;
        color: {TEXT_COLOR};
        font-size: 0.93rem;
        line-height: 1.6;
        margin-bottom: 1rem;
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


def _render_fact_table(tbl: dict):
    name    = tbl.get("name", "unnamed")
    sources = ", ".join(tbl.get("source_tables", []))
    measures= tbl.get("measures", [])
    fks     = tbl.get("foreign_keys", [])
    why     = tbl.get("rationale", "")

    st.markdown(f"""
    <div class="table-card">
        <span class="fact-badge">⚡ FACT TABLE</span>
        <strong style="font-size:1rem;">{name}</strong><br>
        <small style="color:{TEXT_COLOR}88;">Source: {sources}</small>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("📐 Measures")
        if measures:
            for m in measures:
                st.markdown(f"- `{m}`")
        else:
            st.markdown("_(none specified)_")
    with c2:
        st.caption("🔗 Foreign Keys to Dimensions")
        if fks:
            for fk in fks:
                st.markdown(f"- `{fk}`")
        else:
            st.markdown("_(none specified)_")

    if why:
        st.caption(f"💡 {why}")


def _render_dim_table(tbl: dict):
    name   = tbl.get("name", "unnamed")
    sources= ", ".join(tbl.get("source_tables", []))
    attrs  = tbl.get("attributes", [])
    pk     = tbl.get("primary_key", "id")
    why    = tbl.get("rationale", "")

    st.markdown(f"""
    <div class="table-card">
        <span class="dim-badge">📦 DIMENSION TABLE</span>
        <strong style="font-size:1rem;">{name}</strong><br>
        <small style="color:{TEXT_COLOR}88;">Source: {sources}</small>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("🏷️ Attributes")
        if attrs:
            for a in attrs:
                st.markdown(f"- `{a}`")
        else:
            st.markdown("_(none specified)_")
    with c2:
        st.caption("🔑 Primary Key")
        st.markdown(f"`{pk}`")

    if why:
        st.caption(f"💡 {why}")


def show():
    _inject_css()
    st.title("🏛️ Data Warehouse Schema Advisor")
    st.caption(
        "Analyzes your connected database schema and proposes a star schema "
        "redesign with fact/dimension tables and ready-to-review DDL. "
        "Nothing is executed automatically — DDL is for review and download only."
    )

    if not get_engine():
        st.warning(
            "⚠️ No active database connection. "
            "Connect first via **SQL Connect**."
        )
        st.stop()

    meta = get_conn_meta()
    st.info(
        f"Connected to **{meta['database']}** "
        f"({meta.get('dialect_label', 'DB')}) — read-only"
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    col_run, col_clear, _ = st.columns([1, 1, 4])
    run_clicked   = col_run.button("🏛️ Analyze Schema", type="primary")
    clear_clicked = col_clear.button("🗑 Clear")

    if clear_clicked:
        st.session_state.pop("wh_advice", None)
        st.rerun()

    if run_clicked:
        with st.spinner("Gemini is analyzing your schema and designing a star schema…"):
            try:
                advice = analyze_schema_for_warehouse()
                st.session_state["wh_advice"] = advice
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if "wh_advice" not in st.session_state:
        st.info(
            "Click **Analyze Schema** to let Gemini propose a star schema "
            "redesign for your connected database."
        )
        st.stop()

    advice: WarehouseAdvice = st.session_state["wh_advice"]

    # ── Warnings ──────────────────────────────────────────────────────────────
    if advice.warnings:
        with st.expander("⚠️ Schema Warnings", expanded=True):
            for w in advice.warnings:
                st.warning(w)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    _metric_card(m1, advice.n_fact, "Fact Tables Proposed")
    _metric_card(m2, advice.n_dim,  "Dimension Tables Proposed")
    _metric_card(m3, advice.n_ddl,  "DDL Statements Generated")

    # ── Overall rationale ─────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="section-header">📐 Star Schema Design Rationale</div>',
        unsafe_allow_html=True,
    )
    if advice.schema_rationale:
        st.markdown(
            f'<div class="rationale-box">{advice.schema_rationale}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No rationale returned.")

    # ── Dimension tables ──────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="section-header">📦 Proposed Dimension Tables</div>',
        unsafe_allow_html=True,
    )
    if advice.dimension_tables:
        for tbl in advice.dimension_tables:
            _render_dim_table(tbl)
    else:
        st.info("No dimension tables proposed.")

    # ── Fact tables ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="section-header">⚡ Proposed Fact Tables</div>',
        unsafe_allow_html=True,
    )
    if advice.fact_tables:
        for tbl in advice.fact_tables:
            _render_fact_table(tbl)
    else:
        st.info("No fact tables proposed.")

    # ── DDL statements ────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="section-header">🛠️ Generated DDL</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Review carefully before running. Dimension tables are listed first "
        "(FK dependency order). Nothing below executes automatically."
    )

    safe_ddl   = [d for d in advice.ddl_statements if ddl_safe_for_display(d)]
    unsafe_ddl = [d for d in advice.ddl_statements if not ddl_safe_for_display(d)]

    if unsafe_ddl:
        st.error(
            f"⚠️ {len(unsafe_ddl)} DDL statement(s) were filtered out "
            "because they contained non-CREATE TABLE operations."
        )

    if safe_ddl:
        full_ddl = "\n\n".join(safe_ddl)
        for i, ddl in enumerate(safe_ddl, 1):
            with st.expander(f"Statement {i} — {ddl.strip().splitlines()[0][:60]}"):
                st.code(ddl, language="sql")

        st.divider()
        st.download_button(
            label="⬇️ Download All DDL (.sql)",
            data=full_ddl.encode("utf-8"),
            file_name=f"{meta['database']}_star_schema.sql",
            mime="text/plain",
            key="wh_ddl_download",
        )
    else:
        st.info("No safe DDL statements to display.")

    # ── Debug ─────────────────────────────────────────────────────────────────
    with st.expander("🐛 Raw Gemini response (debug)", expanded=False):
        st.code(advice.raw_response or "(empty)", language="text")