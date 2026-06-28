"""
Stage I — Visual Query Builder UI.
Form-based SQL builder: column picks, filters, aggregations, joins, ordering.
All execution goes through connections.db_engine.run_query() — Stage H audits it.
"""

import streamlit as st
import pandas as pd
from connections.db_engine import (
    get_engine, get_conn_meta, run_query, list_tables, describe_table,
)
from connections.sql_query_builder import (
    QueryBuilderSpec, AggregationSpec, FilterSpec,
    AGG_FUNCTIONS, ALL_OPS,
    build_query_sql, interpolate_sql, describe_query, validate_spec,
)
from connections.bridge import apply_bridge
from connections.query_store import save_query
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR, TEXT_COLOR,
    BORDER_COLOR, SUCCESS_BG, SUCCESS_COLOR, WARNING_COLOR,
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
    .sql-preview {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-left: 4px solid {WARNING_COLOR};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: {TEXT_COLOR};
        white-space: pre-wrap;
    }}
    .query-desc {{
        background: {SUCCESS_BG};
        color: {SUCCESS_COLOR};
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def _col_names(table: str) -> list[str]:
    """Return column names for a table, cached in session_state."""
    cache_key = f"vqb_cols_{table}"
    if cache_key not in st.session_state:
        try:
            df = describe_table(table)
            st.session_state[cache_key] = df["name"].tolist()
        except Exception:
            st.session_state[cache_key] = []
    return st.session_state[cache_key]


def show():
    _inject_css()
    st.title("🏗️ Visual Query Builder")
    st.caption(
        "Build SQL queries with dropdowns and filters — no SQL typing required. "
        "Every query runs through the governed execution path."
    )

    if not get_engine():
        st.warning("⚠️ No active database connection. Connect first via **SQL Connect**.")
        st.stop()

    meta = get_conn_meta()
    st.info(
        f"Connected to **{meta['database']}** "
        f"({meta.get('dialect_label','DB')}) — read-only"
    )

    tables = list_tables()
    if not tables:
        st.error("No tables found in the connected database.")
        st.stop()

    # ── ① Source Table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">① Source Table</div>',
                unsafe_allow_html=True)

    table = st.selectbox("Select table", options=tables, key="vqb_table")
    all_cols = _col_names(table)

    if not all_cols:
        st.error(f"Could not read columns for table `{table}`.")
        st.stop()

    # Clear column cache if table changed
    if st.session_state.get("vqb_last_table") != table:
        st.session_state["vqb_last_table"] = table
        for k in ["vqb_select_cols", "vqb_group_cols", "vqb_order_col"]:
            st.session_state.pop(k, None)

    # ── ② Column Selection ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">② Select Columns</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        select_cols = st.multiselect(
            "Plain columns (no aggregation)",
            options=all_cols,
            key="vqb_select_cols",
        )
        distinct = st.checkbox("DISTINCT", key="vqb_distinct")

    with col_right:
        st.caption("Aggregations")
        n_aggs = st.number_input(
            "Number of aggregations", min_value=0, max_value=6,
            value=0, step=1, key="vqb_n_aggs",
        )

    aggregations: list[AggregationSpec] = []
    if n_aggs > 0:
        for i in range(int(n_aggs)):
            ca, cb, cc = st.columns(3)
            func   = ca.selectbox(f"Function #{i+1}", AGG_FUNCTIONS,
                                   key=f"vqb_agg_func_{i}")
            acol   = cb.selectbox(f"Column #{i+1}", all_cols,
                                   key=f"vqb_agg_col_{i}")
            alias  = cc.text_input(f"Alias #{i+1}",
                                    value=f"{func.lower().replace(' ','_')}_{acol}",
                                    key=f"vqb_agg_alias_{i}")
            if acol and alias:
                aggregations.append(AggregationSpec(func=func, column=acol, alias=alias))

    # ── ③ JOIN (optional) ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">③ JOIN (optional)</div>',
                unsafe_allow_html=True)

    use_join = st.checkbox("Add a JOIN", key="vqb_use_join")
    join_table = join_type = join_left = join_right = None

    if use_join:
        other_tables = [t for t in tables if t != table]
        if not other_tables:
            st.warning("No other tables available to JOIN.")
        else:
            j1, j2, j3, j4 = st.columns(4)
            join_type  = j1.selectbox("JOIN type", ["INNER", "LEFT", "RIGHT"],
                                       key="vqb_join_type")
            join_table = j2.selectbox("Join table", other_tables,
                                       key="vqb_join_table")
            join_cols  = _col_names(join_table) if join_table else []
            join_left  = j3.selectbox("Left ON col", all_cols,
                                       key="vqb_join_left")
            join_right = j4.selectbox("Right ON col", join_cols or all_cols,
                                       key="vqb_join_right")

    # ── ④ Filters (WHERE) ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">④ Filters (WHERE)</div>',
                unsafe_allow_html=True)

    n_filters = st.number_input(
        "Number of filters", min_value=0, max_value=8,
        value=0, step=1, key="vqb_n_filters",
    )

    filters: list[FilterSpec] = []
    for i in range(int(n_filters)):
        fc, fo, fv = st.columns(3)
        fcol = fc.selectbox(f"Column #{i+1}", all_cols, key=f"vqb_f_col_{i}")
        fop  = fo.selectbox(f"Operator #{i+1}", ALL_OPS + ["IS NULL", "IS NOT NULL"],
                             key=f"vqb_f_op_{i}")
        fval = ""
        if fop not in ("IS NULL", "IS NOT NULL"):
            fval = fv.text_input(f"Value #{i+1}", key=f"vqb_f_val_{i}")
        filters.append(FilterSpec(column=fcol, operator=fop, value=fval))

    # ── ⑤ Group By ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⑤ Group By</div>',
                unsafe_allow_html=True)

    group_by_cols = st.multiselect(
        "GROUP BY columns (required when aggregations are used with plain columns)",
        options=all_cols,
        key="vqb_group_cols",
    )

    # ── ⑥ HAVING filters ─────────────────────────────────────────────────────
    having_filters: list[FilterSpec] = []
    if aggregations:
        st.markdown('<div class="section-header">⑥ HAVING (post-aggregation filter)</div>',
                    unsafe_allow_html=True)
        agg_aliases = [a.alias for a in aggregations]
        n_having = st.number_input(
            "Number of HAVING conditions", min_value=0, max_value=4,
            value=0, step=1, key="vqb_n_having",
        )
        for i in range(int(n_having)):
            hc, ho, hv = st.columns(3)
            hcol = hc.selectbox(f"Alias #{i+1}", agg_aliases, key=f"vqb_h_col_{i}")
            hop  = ho.selectbox(f"Operator #{i+1}", ["=","!=",">",">=","<","<="],
                                 key=f"vqb_h_op_{i}")
            hval = hv.text_input(f"Value #{i+1}", key=f"vqb_h_val_{i}")
            having_filters.append(FilterSpec(column=hcol, operator=hop, value=hval))

    # ── ⑦ Order By + Limit ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">⑦ Order & Limit</div>',
                unsafe_allow_html=True)

    order_col_options = all_cols + [a.alias for a in aggregations]
    oc1, oc2, oc3 = st.columns(3)
    order_col = oc1.selectbox("Order by", ["(none)"] + order_col_options,
                               key="vqb_order_col")
    order_dir = oc2.selectbox("Direction", ["ASC", "DESC"], key="vqb_order_dir")
    row_limit = oc3.number_input("Row limit", min_value=1, max_value=100_000,
                                  value=1000, step=100, key="vqb_limit")

    order_by = [(order_col, order_dir)] if order_col != "(none)" else []

    # ── Build spec & preview SQL ──────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">⑧ SQL Preview</div>',
                unsafe_allow_html=True)

    spec = QueryBuilderSpec(
        table         = table,
        select_cols   = select_cols,
        aggregations  = aggregations,
        group_by_cols = group_by_cols,
        filters       = filters,
        having_filters= having_filters,
        order_by      = order_by,
        limit         = int(row_limit),
        distinct      = distinct,
        join_table    = join_table,
        join_type     = join_type or "INNER",
        join_on_left  = join_left,
        join_on_right = join_right,
    )

    errors = validate_spec(spec)
    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    try:
        sql, params = build_query_sql(spec)
        display_sql = interpolate_sql(sql, params)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.markdown(
        f'<div class="query-desc">📋 {describe_query(spec)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sql-preview">{display_sql}</div>',
        unsafe_allow_html=True,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    st.caption(
        "⚠️ Review the query above before running. "
        "This is a read-only SELECT — nothing is modified."
    )

    col_run, col_load, col_save, _ = st.columns([1, 1, 1, 3])
    run_clicked  = col_run.button("▶ Run Query", type="primary")
    load_clicked = col_load.button("📥 Load into AnalyticaOS")
    save_clicked = col_save.button("💾 Save Query")

    if run_clicked:
        with st.spinner("Executing…"):
            try:
                df_result = run_query(display_sql)
                st.session_state["vqb_result"] = df_result
                st.session_state["vqb_last_sql"] = display_sql
                st.success(
                    f"✅ {len(df_result):,} rows × {len(df_result.columns)} columns."
                )
            except Exception as e:
                st.error(f"Query error: {e}")

    if "vqb_result" in st.session_state:
        df = st.session_state["vqb_result"]
        st.dataframe(df.head(500).astype(str), use_container_width=True)
        st.caption(f"Preview: first 500 rows. Full result: {len(df):,} rows.")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", data=csv,
            file_name="vqb_result.csv", mime="text/csv",
            key="vqb_csv_dl",
        )

    if load_clicked:
        if "vqb_result" not in st.session_state:
            st.warning("Run the query first before loading.")
        else:
            df_cast, br = apply_bridge(st.session_state["vqb_result"])
            st.session_state["df"]          = df_cast
            st.session_state["data_source"] = f"Visual Query Builder ({meta.get('dialect_label','')})"
            st.success("✅ Loaded into AnalyticaOS.")
            st.caption(br.summary)

    if save_clicked:
        st.session_state["vqb_show_save"] = True

    if st.session_state.get("vqb_show_save"):
        with st.form("vqb_save_form"):
            sn, sd = st.columns(2)
            qname = sn.text_input("Query name", key="vqb_save_name")
            qdesc = sd.text_input("Description (optional)", key="vqb_save_desc")
            if st.form_submit_button("Save"):
                if not qname.strip():
                    st.error("Name required.")
                else:
                    try:
                        save_query(
                            name=qname.strip(),
                            sql=display_sql,
                            description=qdesc.strip(),
                            dialect=meta.get("dialect", "mysql"),
                            source_page="Visual Query Builder",
                        )
                        st.success(f"✅ Saved as **{qname.strip()}**.")
                        st.session_state.pop("vqb_show_save", None)
                    except Exception as e:
                        st.error(f"Save failed: {e}")