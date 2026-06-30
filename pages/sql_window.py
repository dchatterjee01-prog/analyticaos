"""
Stage H + I + J + K — SQL Builders + Saved Queries + SQL Analytics Agent.
Four tabs: Window Functions · CTE Builder · Saved Queries · DB Analytics.
Requires an active DB connection from pages/sql_connect.py.
"""
import pandas as pd
import streamlit as st
from connections.bridge import apply_bridge
from connections.db_engine import (
    get_engine,
    get_conn_meta,
    list_tables,
    describe_table,
    run_query,
)
from connections.sql_builders import (
    WindowFunctionSpec,
    build_window_sql,
    describe_window_function,
    WINDOW_FUNCTIONS,
    _NEEDS_VALUE_COL,
    _NEEDS_OFFSET,
    CTEBlock,
    CTESpec,
    build_cte_sql,
    cte_chaining_hint,
)
from connections.query_store import (
    save_query,
    list_queries,
    delete_query,
    export_queries_sql,
)
from agents.sql_analytics_agent import SQLAnalyticsAgent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SQL Builders", page_icon="🪟", layout="wide")
st.title("🪟 SQL Builders")
st.caption("Window Functions · CTE Builder · Saved Queries · DB Analytics")

# ── Guard: must be connected ──────────────────────────────────────────────────
if not get_engine():
    st.warning(
        "⚠️ No active database connection. "
        "Please connect first via **SQL Connect**."
    )
    st.stop()

meta = get_conn_meta()
st.info(
    f"Connected to **{meta['database']}** "
    f"({meta.get('dialect_label', 'DB')}) — read-only"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_wf, tab_cte, tab_saved, tab_agent = st.tabs([
    "🪟 Window Functions",
    "🔗 CTE Builder",
    "💾 Saved Queries",
    "🤖 DB Analytics",
])


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — inline save form
# ══════════════════════════════════════════════════════════════════════════════
def _render_save_form(sql: str, source_page: str, key_prefix: str) -> None:
    with st.expander("💾 Save this query", expanded=False):
        col_n, col_d = st.columns([2, 3])
        qname = col_n.text_input("Query name", key=f"{key_prefix}_save_name")
        qdesc = col_d.text_input(
            "Description (optional)", key=f"{key_prefix}_save_desc"
        )
        if st.button("Save", key=f"{key_prefix}_save_btn"):
            if not qname.strip():
                st.error("Query name is required.")
            else:
                try:
                    save_query(
                        name=qname.strip(),
                        sql=sql,
                        description=qdesc.strip(),
                        dialect=meta.get("dialect", "mysql"),
                        source_page=source_page,
                    )
                    st.success(f"✅ Saved as **{qname.strip()}**.")
                except Exception as e:
                    st.error(f"Save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WINDOW FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_wf:
    st.subheader("Window Functions Builder")
    st.caption("Build ROW_NUMBER, RANK, LAG, SUM OVER and more — point and click.")

    tables = list_tables()
    if not tables:
        st.error("No tables found in this database.")
        st.stop()

    st.markdown("**① Select Table**")
    selected_table = st.selectbox("Table", tables, key="wf_table")

    try:
        desc_df  = describe_table(selected_table)
        all_cols = desc_df["name"].tolist()
    except Exception as e:
        st.error(f"Could not load columns: {e}")
        st.stop()

    if not all_cols:
        st.error("No columns found for this table.")
        st.stop()

    st.divider()
    st.markdown("**② Choose Window Function**")
    col_func, col_desc = st.columns([1, 2])
    with col_func:
        selected_func = st.selectbox("Function", WINDOW_FUNCTIONS, key="wf_func")
    with col_desc:
        st.markdown(" ")
        st.markdown(" ")
        st.info(describe_window_function(selected_func))

    st.divider()
    st.markdown("**③ Value Column**")
    value_col = None
    if selected_func in _NEEDS_VALUE_COL:
        value_col = st.selectbox(
            f"Column for {selected_func}()", all_cols, key="wf_value_col"
        )
    else:
        st.caption(f"{selected_func}() does not require a value column.")

    st.divider()
    st.markdown("**④ PARTITION BY** *(optional)*")
    partition_cols = st.multiselect(
        "Partition columns", options=all_cols, key="wf_partition"
    )

    st.divider()
    st.markdown("**⑤ ORDER BY**")
    order_col = st.selectbox("Order by column", all_cols, key="wf_order_col")
    order_dir = st.radio(
        "Direction", ["ASC", "DESC"], horizontal=True, key="wf_order_dir"
    )
    order_cols = [(order_col, order_dir)]

    offset      = 1
    default_val = "NULL"
    if selected_func in _NEEDS_OFFSET:
        st.divider()
        st.markdown("**⑥ Offset & Default** *(LEAD/LAG)*")
        col_off, col_def = st.columns(2)
        offset      = col_off.number_input(
            "Offset (rows)", min_value=1, value=1, step=1, key="wf_offset"
        )
        default_val = col_def.text_input(
            "Default value", value="NULL", key="wf_default"
        )

    st.divider()
    st.markdown("**⑦ Additional Columns & Alias**")
    extra_cols = st.multiselect(
        "Extra columns to include in SELECT",
        options=all_cols,
        default=all_cols[:3],
        key="wf_extra_cols",
    )
    result_alias = st.text_input(
        "Result column alias",
        value=f"{selected_func.lower()}_result",
        key="wf_alias",
    )

    st.divider()
    st.markdown("**⑧ Generated SQL**")
    try:
        spec = WindowFunctionSpec(
            table          = selected_table,
            func           = selected_func,
            value_col      = value_col,
            partition_cols = partition_cols,
            order_cols     = order_cols,
            offset         = int(offset),
            default_val    = default_val or "NULL",
            result_alias   = result_alias or f"{selected_func.lower()}_result",
            extra_cols     = extra_cols,
        )
        generated_sql = build_window_sql(spec)
    except Exception as e:
        st.error(f"SQL generation error: {e}")
        st.stop()

    edited_sql = st.text_area(
        "Review / edit before running",
        value=generated_sql,
        height=180,
        key="wf_sql_editor",
    )

    col_run, col_load, _ = st.columns([1, 1, 4])
    wf_run  = col_run.button("▶ Run Query",            key="wf_run")
    wf_load = col_load.button("📥 Load into AnalyticaOS", key="wf_load")

    if wf_run:
        with st.spinner("Executing…"):
            try:
                df_result = run_query(edited_sql)
                st.session_state["wf_result_df"] = df_result
                st.success(
                    f"✅ {len(df_result):,} rows × "
                    f"{len(df_result.columns)} columns returned."
                )
            except Exception as e:
                st.error(f"Query error: {e}")

    if "wf_result_df" in st.session_state:
        df_preview = st.session_state["wf_result_df"]
        st.dataframe(df_preview.head(500).astype(str), use_container_width=True)
        st.caption(
            f"Preview: first 500 rows. "
            f"Full result: {len(df_preview):,} rows."
        )

    if wf_load:
        if "wf_result_df" not in st.session_state:
            st.warning("Run the query first before loading.")
        else:
            df_cast, br = apply_bridge(st.session_state["wf_result_df"])
            st.session_state["df"] = df_cast
            st.session_state["data_source"] = (
                f"Window Function · {selected_func} · {selected_table}"
            )
            st.success("✅ Data loaded into AnalyticaOS.")
            st.caption(br.summary)
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CTE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_cte:
    st.subheader("CTE Builder")
    st.caption("Build WITH … SELECT queries including chained and recursive CTEs.")

    is_recursive = st.toggle(
        "Enable RECURSIVE (WITH RECURSIVE — MySQL 8+, PostgreSQL, SQLite 3.8.3+)",
        value=False,
        key="cte_recursive",
    )
    if is_recursive:
        st.warning(
            "⚠️ Recursive CTEs require MySQL 8.0+, PostgreSQL, or SQLite "
            "3.8.3+. Each recursive block must contain UNION or UNION ALL."
        )

    st.divider()
    st.markdown("**① Define CTE Blocks**")
    num_blocks = st.number_input(
        "Number of CTE blocks", min_value=1, max_value=8,
        value=1, step=1, key="cte_num_blocks",
    )

    blocks:      list[CTEBlock] = []
    block_names: list[str]      = []

    for i in range(int(num_blocks)):
        st.markdown(f"**CTE Block {i + 1}**")
        col_name, col_rec = st.columns([3, 1])
        block_name = col_name.text_input(
            "CTE name (alias)", value=f"cte_{i + 1}", key=f"cte_name_{i}"
        )
        block_recursive = col_rec.checkbox(
            "Recursive", value=False,
            key=f"cte_block_rec_{i}", disabled=not is_recursive,
        )
        placeholder = (
            "SELECT id, name, manager_id, 0 AS depth\n"
            "FROM employees WHERE manager_id IS NULL\n"
            "UNION ALL\n"
            "SELECT e.id, e.name, e.manager_id, t.depth + 1\n"
            "FROM employees e\n"
            f"JOIN {block_name} t ON e.manager_id = t.id"
        ) if block_recursive else "SELECT *\nFROM your_table\nWHERE some_condition"

        block_sql = st.text_area(
            f"SQL for `{block_name}`",
            value=st.session_state.get(f"cte_sql_{i}", placeholder),
            height=140,
            key=f"cte_sql_{i}",
        )
        blocks.append(CTEBlock(
            name=block_name.strip(),
            sql=block_sql.strip(),
            is_recursive=block_recursive,
        ))
        block_names.append(block_name.strip())

    hint = cte_chaining_hint(block_names)
    if hint:
        st.info(hint)

    st.divider()
    st.markdown("**② Final SELECT**")
    st.caption(
        f"Available CTEs: {', '.join(f'`{n}`' for n in block_names)}"
    )
    final_select = st.text_area(
        "Final SELECT",
        value=st.session_state.get(
            "cte_final_select",
            f"SELECT *\nFROM `{block_names[0]}`" if block_names else "SELECT *",
        ),
        height=120,
        key="cte_final_select",
    )

    st.divider()
    st.markdown("**③ Generated SQL**")
    try:
        cte_spec = CTESpec(
            blocks=blocks, final_select=final_select, recursive=is_recursive
        )
        cte_sql = build_cte_sql(cte_spec)
    except Exception as e:
        st.error(f"CTE generation error: {e}")
        st.stop()

    cte_edited_sql = st.text_area(
        "Review / edit before running",
        value=cte_sql,
        height=220,
        key="cte_sql_editor",
    )

    col_run2, col_load2, _ = st.columns([1, 1, 4])
    cte_run  = col_run2.button("▶ Run CTE Query",         key="cte_run")
    cte_load = col_load2.button("📥 Load into AnalyticaOS", key="cte_load")

    if cte_run:
        with st.spinner("Executing…"):
            try:
                df_cte = run_query(cte_edited_sql)
                st.session_state["cte_result_df"] = df_cte
                st.success(
                    f"✅ {len(df_cte):,} rows × "
                    f"{len(df_cte.columns)} columns returned."
                )
            except Exception as e:
                st.error(f"Query error: {e}")

    if "cte_result_df" in st.session_state:
        df_cte_preview = st.session_state["cte_result_df"]
        st.dataframe(df_cte_preview.head(500).astype(str), use_container_width=True)
        st.caption(
            f"Preview: first 500 rows. "
            f"Full result: {len(df_cte_preview):,} rows."
        )

    if cte_load:
        if "cte_result_df" not in st.session_state:
            st.warning("Run the CTE query first before loading.")
        else:
            df_cast, br = apply_bridge(st.session_state["cte_result_df"])
            st.session_state["df"] = df_cast
            st.session_state["data_source"] = "CTE Builder (SQL)"
            st.success("✅ Data loaded into AnalyticaOS.")
            st.caption(br.summary)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SAVED QUERIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_saved:
    st.subheader("💾 Saved Queries Library")
    st.caption("All queries are private to your session identity.")

    col_refresh, col_export, _ = st.columns([1, 1, 4])
    if col_refresh.button("🔄 Refresh", key="sq_refresh"):
        st.rerun()

    if col_export.button("⬇️ Export all as SQL", key="sq_export"):
        export_text = export_queries_sql()
        st.download_button(
            label="Download .sql",
            data=export_text,
            file_name="analyticaos_saved_queries.sql",
            mime="text/plain",
            key="sq_download",
        )

    queries = list_queries()
    if not queries:
        st.info(
            "No saved queries yet. "
            "Use the 💾 Save form in the Window or CTE tabs."
        )
    else:
        st.caption(f"{len(queries)} saved query/queries.")
        for q in queries:
            with st.expander(
                f"**{q.name}** · {q.dialect.upper()} · "
                f"from *{q.source_page}* · saved {q.updated_at[:10]}",
                expanded=False,
            ):
                if q.description:
                    st.caption(q.description)
                st.code(q.sql, language="sql")

                col_load_q, col_run_q, col_del, _ = st.columns([1, 1, 1, 3])

                if col_load_q.button(
                    "📋 Load to query boxes", key=f"sq_copy_{q.id}"
                ):
                    st.session_state["sql_last_query"]   = q.sql
                    st.session_state["nl_generated_sql"] = q.sql
                    st.success(
                        "✅ Loaded into SQL Connect and NL-to-SQL query boxes."
                    )

                if col_run_q.button("▶ Run", key=f"sq_run_{q.id}"):
                    if not get_engine():
                        st.error("No active DB connection.")
                    else:
                        with st.spinner("Executing…"):
                            try:
                                df_sq = run_query(q.sql)
                                st.session_state["sq_result_df"] = df_sq
                                st.session_state["sq_active_id"] = q.id
                                st.success(
                                    f"✅ {len(df_sq):,} rows × "
                                    f"{len(df_sq.columns)} columns."
                                )
                            except Exception as e:
                                st.error(f"Query error: {e}")

                if col_del.button("🗑 Delete", key=f"sq_del_{q.id}"):
                    delete_query(q.id)
                    st.success(f"Deleted **{q.name}**.")
                    st.rerun()

                if (
                    st.session_state.get("sq_active_id") == q.id
                    and "sq_result_df" in st.session_state
                ):
                    df_sq_preview = st.session_state["sq_result_df"]
                    st.dataframe(
                        df_sq_preview.head(500).astype(str),
                        use_container_width=True,
                    )
                    if st.button("📥 Load into AnalyticaOS", key=f"sq_load_{q.id}"):
                        df_cast, br = apply_bridge(df_sq_preview)
                        st.session_state["df"] = df_cast
                        st.session_state["data_source"] = f"Saved Query · {q.name}"
                        st.success("✅ Data loaded into AnalyticaOS.")
                        st.caption(br.summary)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DB ANALYTICS AGENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    st.subheader("🤖 DB Analytics Agent")
    st.caption(
        "Full-database scan — row counts, null rates, "
        "cardinality, top-N values, numeric stats."
    )

    # Cache result in session_state to avoid re-scanning on every rerun
    _AGENT_KEY = "sql_agent_result"

    col_scan, col_clear, _ = st.columns([1, 1, 4])
    run_agent   = col_scan.button("🔍 Run Full Scan",   key="agent_run")
    clear_agent = col_clear.button("🗑 Clear Results",  key="agent_clear")

    if clear_agent:
        st.session_state.pop(_AGENT_KEY, None)
        st.rerun()

    if run_agent:
        tables_count = len(list_tables())
        with st.spinner(
            f"Scanning {tables_count} table(s)… "
            "this may take a moment on large databases."
        ):
            agent  = SQLAnalyticsAgent()
            result = agent.run()
            st.session_state[_AGENT_KEY] = result

    # ── Render cached result ──────────────────────────────────────────────────
    if _AGENT_KEY not in st.session_state:
        st.info("Click **Run Full Scan** to analyse all tables in the connected database.")
        st.stop()

    result = st.session_state[_AGENT_KEY]

    # Status banner
    if result.status == "error":
        st.error(result.summary)
        st.stop()

    st.success(result.summary)

    # ── Cross-table summary DataFrame ─────────────────────────────────────────
    st.divider()
    st.subheader("📊 Table Summary")
    summary_df = result.artifacts.get("table_summary")
    if summary_df is not None and not summary_df.empty:
        st.dataframe(summary_df.astype(str), use_container_width=True)

    # ── Findings ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔎 Findings")

    findings = result.findings
    if not findings:
        st.success("No issues found across all tables.")
    else:
        high = [f for f in findings if f["severity"] == "high"]
        med  = [f for f in findings if f["severity"] == "medium"]
        low  = [f for f in findings if f["severity"] == "low"]

        col_h, col_m, col_l = st.columns(3)
        col_h.metric("🔴 High",   len(high))
        col_m.metric("🟡 Medium", len(med))
        col_l.metric("🟢 Low",    len(low))

        for severity, items, colour in [
            ("high",   high, "🔴"),
            ("medium", med,  "🟡"),
            ("low",    low,  "🟢"),
        ]:
            if items:
                with st.expander(
                    f"{colour} {severity.capitalize()} ({len(items)})",
                    expanded=(severity == "high"),
                ):
                    for f in items:
                        st.markdown(f"- {f['message']}")

    # ── Recommendations ───────────────────────────────────────────────────────
    if result.recommendations:
        st.divider()
        st.subheader("💡 Recommendations")
        for rec in result.recommendations:
            st.markdown(f"- {rec}")

    # ── Per-table column profiles ─────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Per-Table Column Profiles")

    profiles = result.artifacts.get("profiles", [])
    for profile in profiles:
        with st.expander(
            f"**{profile.name}** — "
            f"{profile.row_count:,} rows · {profile.col_count} columns",
            expanded=False,
        ):
            if profile.warnings:
                for w in profile.warnings:
                    st.warning(w)
                continue

            col_data = []
            for c in profile.columns:
                col_data.append({
                    "column":    c.name,
                    "type":      c.dtype,
                    "nulls":     f"{c.null_count:,} ({c.null_rate:.0%})",
                    "distinct":  f"{c.distinct_count:,} ({c.cardinality:.0%})",
                    "min":       c.min_val,
                    "max":       c.max_val,
                    "mean":      c.mean_val,
                    "top values": ", ".join(
                        str(tv.get("value", "")) for tv in c.top_values[:3]
                    ),
                })

            if col_data:
                st.dataframe(
                    pd.DataFrame(col_data).astype(str),
                    use_container_width=True,
                )

    # ── Load summary into AnalyticaOS ─────────────────────────────────────────
    st.divider()
    if st.button("📥 Load Table Summary into AnalyticaOS", key="agent_load"):
        if summary_df is not None and not summary_df.empty:
            df_cast, br = apply_bridge(summary_df)
            st.session_state["df"] = df_cast
            st.session_state["data_source"] = "SQL Analytics Agent — Table Summary"
            st.success("✅ Table summary loaded into AnalyticaOS.")
            st.caption(br.summary)
        else:
            st.warning("No summary data to load.")