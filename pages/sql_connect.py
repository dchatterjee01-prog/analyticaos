"""
Stage F — SQL Connect page.
Multi-dialect entry point: MySQL, PostgreSQL, SQLite, SQL Server.
Populates session_state["sql_engine"] and session_state["df"].
"""

import streamlit as st
from connections.db_engine import (
    build_engine,
    get_engine,
    get_conn_meta,
    drop_engine,
    run_query,
    list_tables,
    describe_table,
    table_row_count,
    DIALECT_REGISTRY,
)

_HISTORY_KEY = "sql_query_history"
_HISTORY_MAX = 10

# Dialects that need host/port/user/pass (SQLite only needs a file path)
_NEEDS_CREDENTIALS = {"mysql", "postgresql", "mssql"}


def _push_history(sql: str) -> None:
    history: list = st.session_state.get(_HISTORY_KEY, [])
    history = [q for q in history if q.strip() != sql.strip()]
    history.insert(0, sql.strip())
    st.session_state[_HISTORY_KEY] = history[:_HISTORY_MAX]


st.set_page_config(page_title="SQL Connect", page_icon="🔌", layout="wide")
st.title("🔌 SQL Connect")
st.caption("Credentials are session-only and never saved to disk.")

# ── Already connected ─────────────────────────────────────────────────────────
if get_engine():
    meta = get_conn_meta()
    dialect_label = meta.get("dialect_label", "Database")

    st.success(
        f"✅ Connected to **{meta['database']}** "
        f"({dialect_label}) on `{meta['host']}:{meta['port']}` "
        f"as `{meta['username']}` — **read-only**"
        if meta.get("dialect") != "sqlite"
        else f"✅ Connected to SQLite file **{meta['database']}** — **read-only**"
    )

    if st.button("Disconnect"):
        drop_engine()
        st.rerun()

    # ── Schema Browser ────────────────────────────────────────────────────────
    st.divider()
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
                    st.caption(
                        f"**{selected_table}** — "
                        f"~{approx:,} rows (estimate)"
                    )
                    # Normalised columns: name / type / nullable / default / primary_key
                    st.dataframe(desc_df.astype(str), use_container_width=True)

                    # Scaffold uses normalised "name" column
                    col_names = desc_df["name"].tolist()[:10]
                    cols      = ", ".join(col_names)
                    scaffold  = (
                        f"SELECT {cols}\nFROM `{selected_table}`\nLIMIT 100;"
                    )
                    if st.button("📋 Scaffold SELECT into query box"):
                        st.session_state["sql_last_query"] = scaffold
                        st.rerun()
                except Exception as e:
                    st.error(f"Could not describe table: {e}")

    # ── Query Runner ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚡ Run Query")

    history: list = st.session_state.get(_HISTORY_KEY, [])
    if history:
        history_options = ["— select a past query —"] + history
        chosen = st.selectbox(
            "📜 Query history (last 10)",
            options=history_options,
            key="sql_history_select",
        )
        if chosen != "— select a past query —":
            if st.button("↩ Load selected into query box"):
                st.session_state["sql_last_query"] = chosen
                st.rerun()

    default_sql = "SELECT *\nFROM your_table\nLIMIT 100;"
    sql_input = st.text_area(
        "SQL (SELECT / WITH only)",
        value=st.session_state.get("sql_last_query", default_sql),
        height=160,
        key="sql_query_input",
    )

    col_run, col_load, col_clear, _ = st.columns([1, 1, 1, 3])
    run_clicked   = col_run.button("▶ Run Query")
    load_clicked  = col_load.button("📥 Load into AnalyticaOS")
    clear_clicked = col_clear.button("🗑 Clear History")

    if clear_clicked:
        st.session_state[_HISTORY_KEY] = []
        st.rerun()

    if run_clicked:
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
        st.dataframe(
            df_preview.head(500).astype(str),
            use_container_width=True,
        )
        st.caption(
            f"Preview: first 500 rows shown. "
            f"Full result: {len(df_preview):,} rows."
        )

    if load_clicked:
        if "sql_result_df" not in st.session_state:
            st.warning("Run a query first before loading.")
        else:
            st.session_state["df"] = st.session_state["sql_result_df"].copy()
            st.session_state["data_source"] = (
                f"SQL Query ({meta.get('dialect_label', 'Database')})"
            )
            st.success(
                "✅ Data loaded into AnalyticaOS. "
                "All downstream pages (EDA, ML, Forecast…) will now use this dataset."
            )

    st.stop()

# ── Not yet connected — show connect form ─────────────────────────────────────
st.warning(
    "⚠️ Credentials exist in this session only. "
    "They are lost on refresh or logout."
)

# Dialect selector — drives which fields are shown
dialect_options = {v["label"]: k for k, v in DIALECT_REGISTRY.items()}
dialect_label   = st.selectbox(
    "Database type",
    options=list(dialect_options.keys()),
    key="connect_dialect_select",
)
dialect = dialect_options[dialect_label]

with st.form("db_connect_form"):
    if dialect == "sqlite":
        database  = st.text_input(
            "SQLite file path",
            placeholder="C:/data/mydb.db or relative/path.db",
        )
        host      = "localhost"
        port      = 0
        username  = ""
        password  = ""
    else:
        col1, col2 = st.columns([3, 1])
        host      = col1.text_input("Host", value="localhost")
        port      = col2.number_input("Port", value=_default_port(dialect), step=1)
        database  = st.text_input("Database name")
        username  = st.text_input("Username")
        password  = st.text_input("Password", type="password")

    submitted = st.form_submit_button(f"Connect to {dialect_label} (read-only)")


def _default_port(d: str) -> int:
    return {"mysql": 3306, "postgresql": 5432, "mssql": 1433}.get(d, 3306)


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