"""
Stage F + L + M + G — SQL Connect page.
Tabs: Query Runner · Optimize · Export
Bridge: auto dtype-cast on every session_state["df"] assignment.
Stage G: Optimize tab runs EXPLAIN / EXPLAIN ANALYZE via connections.sql_optimizer
(PostgreSQL, MySQL, SQLite). EXPLAIN ANALYZE actually executes the query, so it
passes through the same sql_guard check as the Query Runner tab.
"""

import os
import tempfile
import uuid

import pandas as pd
import streamlit as st
from sqlalchemy import inspect
from connections.db_engine import (
    build_engine, get_engine, get_conn_meta, drop_engine,
    run_query, list_tables, describe_table, table_row_count,
    DIALECT_REGISTRY,
)
from connections.exporter import export_bytes, export_summary, EXPORT_FORMATS
from connections.bridge import apply_bridge
from connections.sql_guard import check_sql, guard_summary
from connections.sql_optimizer import optimize_query, SUPPORTED_DIALECTS

_HISTORY_KEY       = "sql_query_history"
_HISTORY_MAX       = 10
_NEEDS_CREDENTIALS = {"mysql", "postgresql", "mssql"}
_RESULT_SOURCES    = {
    "SQL Connect — last query":      "sql_result_df",
    "NL-to-SQL — last result":       "nl_result_df",
    "Window Function — last result": "wf_result_df",
    "CTE Builder — last result":     "cte_result_df",
    "DB Analytics — table summary":  "sq_result_df",
}

_SEVERITY_RENDER = {
    "high":   ("🔴", "error"),
    "medium": ("🟡", "warning"),
    "low":    ("⚪", "info"),
}


def _push_history(sql: str) -> None:
    history: list = st.session_state.get(_HISTORY_KEY, [])
    history = [q for q in history if q.strip() != sql.strip()]
    history.insert(0, sql.strip())
    st.session_state[_HISTORY_KEY] = history[:_HISTORY_MAX]


def _default_port(d: str) -> int:
    return {"mysql": 3306, "postgresql": 5432, "mssql": 1433}.get(d, 3306)


def show():
    st.set_page_config(page_title="SQL Connect", page_icon="🔌", layout="wide")
    st.title("🔌 SQL Connect")
    st.caption("Credentials are session-only and never saved to disk.")

    # ── Not yet connected ─────────────────────────────────────────────────────────
    if not get_engine():
        st.warning(
            "⚠️ Credentials exist in this session only. "
            "They are lost on refresh or logout."
        )

        # ── One-click demo database (no setup required) ────────────────────
        st.markdown("### Try it instantly")
        st.caption(
            "No database? Spin up a free, private SQLite database in one "
            "click — perfect for testing every SQL Engine page with your "
            "own CSV/Excel data."
        )
        if st.button(
            "✨ Create a new database (no setup required)",
            type="primary",
            width="stretch",
            key="connect_quick_sqlite_btn",
        ):
            session_id = st.session_state.get("_analyticaos_sqlite_session_id")
            if not session_id:
                session_id = uuid.uuid4().hex
                st.session_state["_analyticaos_sqlite_session_id"] = session_id
            temp_db_path = os.path.join(
                tempfile.gettempdir(), f"analyticaos_{session_id}.db"
            )
            with st.spinner("Creating your database…"):
                try:
                    build_engine(
                        host="localhost", port=0, database=temp_db_path,
                        username="", password="", dialect="sqlite",
                    )
                    st.success("New SQLite database created — you're connected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not create database: {e}")

        st.divider()
        st.markdown("### Or connect to your own database")

        dialect_options = {v["label"]: k for k, v in DIALECT_REGISTRY.items()}
        dialect_label   = st.selectbox(
            "Database type", options=list(dialect_options.keys()),
            key="connect_dialect_select",
        )
        dialect = dialect_options[dialect_label]

        with st.form("db_connect_form"):
            if dialect == "sqlite":
                database = st.text_input(
                    "SQLite file path",
                    placeholder="C:/data/mydb.db",
                )
                host = "localhost"; port = 0; username = ""; password = ""
            else:
                col1, col2 = st.columns([3, 1])
                host     = col1.text_input("Host", value="localhost")
                port     = col2.number_input("Port", value=_default_port(dialect), step=1)
                database = st.text_input("Database name")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(f"Connect to {dialect_label} (read-only)")

        if submitted:
            missing = not database
            if dialect in _NEEDS_CREDENTIALS:
                missing = missing or not all([host, username, password])
            if missing:
                st.error("All required fields must be filled.")
            else:
                with st.spinner(f"Connecting to {dialect_label}…"):
                    try:
                        build_engine(host, int(port), database, username, password, dialect)
                        st.success("Connected!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
        st.stop()

    # ── Connected ─────────────────────────────────────────────────────────────────
    meta = get_conn_meta()
    st.success(
        f"✅ Connected to **{meta['database']}** "
        f"({meta.get('dialect_label','DB')}) on "
        f"`{meta['host']}:{meta['port']}` as `{meta['username']}` — **read-only**"
        if meta.get("dialect") != "sqlite"
        else f"✅ Connected to SQLite file **{meta['database']}** — **read-only**"
    )
    if st.button("Disconnect"):
        drop_engine()
        st.rerun()

    # ── First-table import for fresh/empty databases ────────────────────────
    _engine_for_check = get_engine()
    _existing_tables  = inspect(_engine_for_check).get_table_names()

    if len(_existing_tables) == 0:
        st.info(
            "This database is empty. Upload a CSV or Excel file to create "
            "your first table."
        )
        uploaded_file = st.file_uploader(
            "Upload data file", type=["csv", "xlsx", "xls"],
            key="sql_connect_first_upload",
        )
        if uploaded_file is not None:
            default_table_name = (
                os.path.splitext(uploaded_file.name)[0]
                .replace(" ", "_").lower()
            )
            new_table_name = st.text_input(
                "Table name", value=default_table_name, key="sql_connect_first_table_name"
            )
            if st.button("Import as table", key="sql_connect_first_import_btn"):
                try:
                    if uploaded_file.name.lower().endswith(".csv"):
                        df_upload = pd.read_csv(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)

                    df_upload.to_sql(
                        new_table_name, _engine_for_check,
                        if_exists="replace", index=False,
                    )
                    st.success(
                        f"Imported {len(df_upload):,} rows into table "
                        f"'{new_table_name}'."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")
        st.divider()

    tab_query, tab_optimize, tab_export = st.tabs(["⚡ Query Runner", "🚀 Optimize", "📤 Export"])

    # ══════════════════════════════════════════════════════════════════════════════
    with tab_query:
        st.subheader("🗂️ Schema Browser")
        tables = list_tables()
        if not tables:
            st.info("No tables found in this database.")
        else:
            col_tbl, col_detail = st.columns([1, 2])
            with col_tbl:
                st.caption(f"{len(tables)} table(s) found")
                selected_table = st.selectbox(
                    "Select a table", tables, key="schema_table_select"
                )
            with col_detail:
                if selected_table:
                    try:
                        desc_df = describe_table(selected_table)
                        approx  = table_row_count(selected_table)
                        st.caption(f"**{selected_table}** — ~{approx:,} rows (estimate)")
                        st.dataframe(desc_df.astype(str), use_container_width=True)
                        col_names = desc_df["name"].tolist()[:10]
                        scaffold  = (
                            f"SELECT {', '.join(col_names)}\n"
                            f"FROM `{selected_table}`\nLIMIT 100;"
                        )
                        if st.button("📋 Scaffold SELECT into query box"):
                            st.session_state["sql_last_query"] = scaffold
                            st.rerun()
                    except Exception as e:
                        st.error(f"Could not describe table: {e}")

        st.divider()
        st.subheader("⚡ Run Query")

        history: list = st.session_state.get(_HISTORY_KEY, [])
        if history:
            chosen = st.selectbox(
                "📜 Query history (last 10)",
                options=["— select a past query —"] + history,
                key="sql_history_select",
            )
            if chosen != "— select a past query —":
                if st.button("↩ Load selected into query box"):
                    st.session_state["sql_last_query"] = chosen
                    st.rerun()

        sql_input = st.text_area(
            "SQL (SELECT / WITH only)",
            value=st.session_state.get("sql_last_query", "SELECT *\nFROM your_table\nLIMIT 100;"),
            height=160,
            key="sql_query_input",
        )

        # Stage G: independent safety scan, re-evaluated live against whatever
        # is currently typed in the box — same two-checkpoint pattern as
        # pages/sql_nlquery.py. This protects against the same run_query()
        # SELECT/WITH-prefix bypass (e.g. a CTE wrapping a DELETE), regardless
        # of whether the SQL was hand-typed here or pasted from elsewhere.
        guard_result = check_sql(sql_input)

        if guard_result.safe:
            st.success(guard_summary(guard_result))
        else:
            st.error(guard_summary(guard_result))
            if guard_result.blocked_clauses:
                with st.expander("🛡️ Safety check details", expanded=True):
                    for clause in guard_result.blocked_clauses:
                        st.markdown(f"- {clause}")

        col_run, col_load, col_clear, _ = st.columns([1, 1, 1, 3])
        run_clicked   = col_run.button("▶ Run Query", disabled=not guard_result.safe)
        load_clicked  = col_load.button("📥 Load into AnalyticaOS")
        clear_clicked = col_clear.button("🗑 Clear History")

        if clear_clicked:
            st.session_state[_HISTORY_KEY] = []
            st.rerun()

        if run_clicked:
            # Defense in depth: re-verify immediately before execution, even
            # though the button is already disabled when unsafe. Never trust
            # only the UI-disabled state — Streamlit reruns top-to-bottom on
            # every interaction, so this is cheap and closes any timing gap.
            final_check = check_sql(sql_input)
            if not final_check.safe:
                st.error(guard_summary(final_check))
            else:
                with st.spinner("Executing…"):
                    try:
                        df_result = run_query(sql_input)
                        _push_history(sql_input)
                        st.session_state["sql_last_query"] = sql_input
                        st.session_state["sql_result_df"]  = df_result
                        st.success(
                            f"✅ {len(df_result):,} rows × "
                            f"{len(df_result.columns)} columns returned."
                        )
                    except Exception as e:
                        st.error(f"Query error: {e}")

        if "sql_result_df" in st.session_state:
            df_preview = st.session_state["sql_result_df"]
            st.dataframe(df_preview.head(500).astype(str), use_container_width=True)
            st.caption(f"Preview: first 500 rows. Full result: {len(df_preview):,} rows.")

        if load_clicked:
            if "sql_result_df" not in st.session_state:
                st.warning("Run a query first before loading.")
            else:
                df_cast, br = apply_bridge(st.session_state["sql_result_df"])
                st.session_state["df"] = df_cast
                st.session_state["data_source"] = (
                    f"SQL Query ({meta.get('dialect_label','Database')})"
                )
                st.success("✅ Data loaded into AnalyticaOS.")
                st.caption(br.summary)

    # ══════════════════════════════════════════════════════════════════════════════
    with tab_optimize:
        st.subheader("🚀 SQL Optimization Agent")
        st.caption(
            "Run EXPLAIN against your connected database to detect missing "
            "indexes, possible Cartesian joins, and other performance issues."
        )

        current_dialect = meta.get("dialect", "")
        if current_dialect not in SUPPORTED_DIALECTS:
            st.warning(
                f"⚠️ Optimization analysis currently supports PostgreSQL and "
                f"MySQL only. The active connection is "
                f"**{meta.get('dialect_label', current_dialect)}**, which isn't "
                f"supported yet."
            )
        else:
            opt_sql_input = st.text_area(
                "SQL to analyze (SELECT / WITH only)",
                value=st.session_state.get(
                    "opt_sql_input", "SELECT *\nFROM your_table\nLIMIT 100;"
                ),
                height=160,
                key="opt_sql_input",
            )

            opt_guard = check_sql(opt_sql_input)
            if opt_guard.safe:
                st.success(guard_summary(opt_guard))
            else:
                st.error(guard_summary(opt_guard))
                if opt_guard.blocked_clauses:
                    with st.expander("🛡️ Safety check details", expanded=True):
                        for clause in opt_guard.blocked_clauses:
                            st.markdown(f"- {clause}")

            is_sqlite = current_dialect == "sqlite"
            analyze_caption = (
                "**EXPLAIN** shows SQLite's scan strategy without running the "
                "query. SQLite has no EXPLAIN ANALYZE equivalent, so that "
                "option is disabled here."
                if is_sqlite else
                "**EXPLAIN** estimates the plan without running the query. "
                "**EXPLAIN ANALYZE** actually executes it to get real timing — "
                "treat it the same as running the query."
            )
            st.caption(analyze_caption)

            col_explain, col_analyze = st.columns(2)
            explain_clicked = col_explain.button(
                "📋 EXPLAIN (no execution)", disabled=not opt_guard.safe
            )
            analyze_clicked = col_analyze.button(
                "▶ EXPLAIN ANALYZE (executes query)",
                disabled=(not opt_guard.safe) or is_sqlite,
            )

            if explain_clicked or analyze_clicked:
                # Defense in depth: re-verify right before either EXPLAIN mode
                # runs, even though both buttons are already disabled when
                # unsafe — same two-checkpoint pattern as the Query Runner tab.
                final_opt_check = check_sql(opt_sql_input)
                if not final_opt_check.safe:
                    st.error(guard_summary(final_opt_check))
                else:
                    analyze_mode = analyze_clicked
                    spinner_msg  = (
                        "Running EXPLAIN ANALYZE (executing query)…"
                        if analyze_mode else "Running EXPLAIN…"
                    )
                    with st.spinner(spinner_msg):
                        try:
                            opt_result = optimize_query(
                                get_engine(), opt_sql_input,
                                dialect=current_dialect, analyze=analyze_mode,
                            )
                            st.session_state["opt_result"] = opt_result
                        except Exception as e:
                            st.error(f"Optimization analysis failed: {e}")

            if "opt_result" in st.session_state:
                opt_result = st.session_state["opt_result"]
                st.divider()

                status_render = {
                    "ok": st.success, "warning": st.warning, "error": st.error,
                }.get(opt_result.status, st.info)
                status_render(f"**{opt_result.summary}**")

                if opt_result.findings:
                    st.markdown("#### Findings")
                    for f in opt_result.findings:
                        icon, _ = _SEVERITY_RENDER.get(f.get("severity", "low"), ("⚪", "info"))
                        st.markdown(f"{icon} **{f['title']}**  \n{f['detail']}")

                if opt_result.recommendations:
                    st.markdown("#### Recommendations")
                    for i, rec in enumerate(opt_result.recommendations, 1):
                        st.info(f"💡 **{i}.** {rec}")

                raw_plan = opt_result.artifacts.get("raw_plan", "")
                if raw_plan:
                    with st.expander("🔍 Raw EXPLAIN output", expanded=False):
                        st.code(raw_plan, language="text")

    # ══════════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.subheader("📤 Export Query Results")
        st.caption("Export any SQL result from this session to CSV or XLSX.")

        st.markdown("**⚡ Quick Export — current SQL Connect result**")
        if "sql_result_df" not in st.session_state:
            st.info("Run a query in the Query Runner tab first.")
        else:
            df_quick = st.session_state["sql_result_df"]
            summ     = export_summary(df_quick)
            st.caption(
                f"{summ['rows']:,} rows · {summ['columns']} columns · "
                f"~{summ['size_kb']} KB"
            )
            col_fmt, col_sheet, _ = st.columns([1, 2, 3])
            fmt_quick   = col_fmt.selectbox("Format", EXPORT_FORMATS, key="export_fmt_quick")
            sheet_quick = col_sheet.text_input(
                "Sheet name (XLSX only)", value="Query Result", key="export_sheet_quick"
            )
            if st.button("⬇️ Download", key="export_quick_btn"):
                try:
                    data, fname, mime = export_bytes(df_quick, fmt=fmt_quick, sheet_name=sheet_quick)
                    st.download_button(
                        label=f"📄 {fname}", data=data,
                        file_name=fname, mime=mime, key="export_quick_dl",
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")

        st.divider()
        st.markdown("**🗂️ Export Any Result from This Session**")
        available = {
            label: key for label, key in _RESULT_SOURCES.items()
            if key in st.session_state
            and isinstance(st.session_state[key], pd.DataFrame)
            and not st.session_state[key].empty
        }
        if not available:
            st.info("No results available yet.")
        else:
            source_label = st.selectbox(
                "Select result to export", options=list(available.keys()),
                key="export_source_select",
            )
            df_picked = st.session_state[available[source_label]]
            summ_p    = export_summary(df_picked)
            st.caption(
                f"**{source_label}** — {summ_p['rows']:,} rows · "
                f"{summ_p['columns']} columns · ~{summ_p['size_kb']} KB"
            )
            st.dataframe(df_picked.head(10).astype(str), use_container_width=True)
            st.caption("Preview: first 10 rows.")

            col_fmt2, col_sheet2, col_idx, _ = st.columns([1, 2, 1, 2])
            fmt_pick   = col_fmt2.selectbox("Format", EXPORT_FORMATS, key="export_fmt_pick")
            sheet_pick = col_sheet2.text_input(
                "Sheet name (XLSX only)", value=source_label[:31], key="export_sheet_pick"
            )
            incl_idx   = col_idx.checkbox("Include index", value=False, key="export_idx_pick")

            if st.button("⬇️ Download", key="export_pick_btn"):
                try:
                    data, fname, mime = export_bytes(
                        df_picked, fmt=fmt_pick,
                        sheet_name=sheet_pick, index=incl_idx,
                    )
                    st.download_button(
                        label=f"📄 {fname}", data=data,
                        file_name=fname, mime=mime, key="export_pick_dl",
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")

            st.divider()
            if st.button("📥 Load selected result into AnalyticaOS", key="export_load_btn"):
                df_cast, br = apply_bridge(df_picked)
                st.session_state["df"] = df_cast
                st.session_state["data_source"] = source_label
                st.success(f"✅ **{source_label}** loaded into AnalyticaOS.")
                st.caption(br.summary)