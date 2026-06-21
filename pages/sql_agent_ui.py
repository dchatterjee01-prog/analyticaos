import streamlit as st
import pandas as pd
from mysql.connector import Error as MySQLError
from sql_agent import get_connection, fetch_schema_summary, generate_sql, run_readonly_query
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR,
    TEXT_COLOR, BACKGROUND_COLOR, BORDER_COLOR, MUTED_COLOR,
    WARNING_COLOR, WARNING_BG, ERROR_COLOR, ERROR_BG,
)


def _inject_css():
    st.markdown(f"""
    <style>
    .section-header {{
        color: {PRIMARY_COLOR};
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 0.6rem;
    }}
    .sql-preview {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-left: 4px solid {WARNING_COLOR};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: {TEXT_COLOR};
        white-space: pre-wrap;
    }}
    .explanation-box {{
        background: {SURFACE_COLOR};
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.4rem;
        color: {TEXT_COLOR};
        font-size: 0.92rem;
    }}
    .conn-status-ok {{
        background: #E8F3EC;
        border: 1px solid #2E7D5B;
        color: #2E7D5B;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .conn-status-bad {{
        background: {ERROR_BG};
        border: 1px solid {ERROR_COLOR};
        color: {ERROR_COLOR};
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)


def _get_or_create_connection():
    """Connects once per session and caches the connection + schema
    summary in session_state, rather than reconnecting on every rerun."""
    if "sql_agent_conn" in st.session_state:
        conn = st.session_state["sql_agent_conn"]
        try:
            conn.ping(reconnect=True, attempts=1, delay=0)
            return conn, None
        except Exception:
            del st.session_state["sql_agent_conn"]

    try:
        conn = get_connection(st.secrets)
        st.session_state["sql_agent_conn"] = conn
        return conn, None
    except MySQLError as e:
        return None, f"MySQL connection failed: {e}"
    except KeyError as e:
        return None, f"Missing credential in .streamlit/secrets.toml: {e}"
    except Exception as e:
        return None, f"Unexpected connection error: {e}"


def show():
    _inject_css()
    st.title("🗄️ SQL Agent")
    st.caption("Stage B — Ask your MySQL database questions in plain English")

    # ── Connection ──
    conn, conn_error = _get_or_create_connection()

    if conn_error:
        st.markdown(
            f'<span class="conn-status-bad">❌ Not connected</span>',
            unsafe_allow_html=True
        )
        st.error(conn_error)
        st.info(
            "Add MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE "
            "as top-level keys in `.streamlit/secrets.toml` (not nested under "
            "any other section)."
        )
        return

    db_name = st.secrets.get("MYSQL_DATABASE", "unknown")
    st.markdown(
        f'<span class="conn-status-ok">✅ Connected to `{db_name}`</span>',
        unsafe_allow_html=True
    )

    if not st.secrets.get("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY not configured in `.streamlit/secrets.toml`.")
        return

    # ── Schema summary (cached per session, refreshable) ──
    if "sql_agent_schema" not in st.session_state:
        with st.spinner("Reading database schema..."):
            st.session_state["sql_agent_schema"] = fetch_schema_summary(conn)

    with st.expander("📋 View database schema"):
        st.code(st.session_state["sql_agent_schema"], language="text")
        if st.button("🔄 Refresh schema"):
            st.session_state["sql_agent_schema"] = fetch_schema_summary(conn)
            st.rerun()

    # ── History ──
    if "sql_agent_history" not in st.session_state:
        st.session_state["sql_agent_history"] = []

    st.markdown('<div class="section-header">Ask a Question</div>', unsafe_allow_html=True)

    question = st.text_input(
        "What do you want to know?",
        placeholder="e.g. Which movie had the highest opening weekend?",
        key="sql_agent_question_input",
    )

    if st.button("🔮 Generate Query", type="primary", width="stretch"):
        if not question.strip():
            st.warning("⚠️ Enter a question first.")
        else:
            with st.spinner("Asking Gemini..."):
                result = generate_sql(
                    question,
                    st.session_state["sql_agent_schema"],
                    st.secrets["GEMINI_API_KEY"],
                )
            st.session_state["sql_agent_pending"] = {
                "question": question,
                "result": result,
            }

    # ── Pending query: show for approval, never auto-execute ──
    pending = st.session_state.get("sql_agent_pending")
    if pending:
        result = pending["result"]

        if "error" in result:
            st.error(f"❌ {result['error']}")
            if "rejected_sql" in result:
                st.markdown(
                    f'<div class="sql-preview">{result["rejected_sql"]}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown('<div class="section-header">Review Before Running</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="explanation-box">💡 {result["explanation"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="sql-preview">{result["sql"]}</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "⚠️ This is a read-only SELECT query. Review it before approving — "
                "nothing runs until you click below."
            )

            col1, col2 = st.columns(2)
            with col1:
                approve = st.button("✅ Approve & Run", type="primary", width="stretch")
            with col2:
                reject = st.button("🚫 Discard", width="stretch")

            if approve:
                try:
                    cols, rows = run_readonly_query(conn, result["sql"])
                    df_result = pd.DataFrame(rows, columns=cols)

                    st.session_state["sql_agent_history"].append({
                        "question": pending["question"],
                        "sql": result["sql"],
                        "explanation": result["explanation"],
                        "row_count": len(df_result),
                    })
                    st.session_state["sql_agent_last_result"] = df_result
                    st.session_state["sql_agent_pending"] = None
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ Execution blocked: {e}")
                except MySQLError as e:
                    st.error(f"❌ MySQL error: {e}")

            if reject:
                st.session_state["sql_agent_pending"] = None
                st.rerun()

    # ── Last result ──
    if "sql_agent_last_result" in st.session_state:
        st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)
        df_result = st.session_state["sql_agent_last_result"]
        st.dataframe(df_result.astype(str), width="stretch")
        st.caption(f"{len(df_result):,} row(s) returned.")

        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results (CSV)",
            data=csv,
            file_name="sql_agent_results.csv",
            mime="text/csv",
            key="dl_sql_results",
        )

        if st.button("📥 Use as Active Dataset", width="stretch"):
            st.session_state["df"] = df_result
            st.session_state["df_original"] = df_result.copy()
            st.session_state["filename"] = "SQL Agent query result"
            st.success("✅ This result is now the active dataset across the app.")

    # ── Query history ──
    if st.session_state["sql_agent_history"]:
        with st.expander(f"🕘 Query History ({len(st.session_state['sql_agent_history'])})"):
            for h in reversed(st.session_state["sql_agent_history"]):
                st.markdown(f"**Q:** {h['question']}")
                st.code(h["sql"], language="sql")
                st.caption(f"{h['row_count']} row(s) returned")
                st.divider()