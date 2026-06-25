"""
Stage G — NL-to-SQL page.
User types a plain-English question → Gemini generates SQL →
run_query executes it → result previewed + loadable into session_state["df"].
Requires an active DB connection from pages/sql_connect.py.
"""

import streamlit as st
import google.generativeai as genai
from connections.db_engine import (
    get_engine,
    get_conn_meta,
    run_query,
    build_schema_context,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NL-to-SQL", page_icon="🧠", layout="wide")
st.title("🧠 NL-to-SQL")
st.caption("Ask a question in plain English — Gemini writes the SQL.")

# ── Guard: must be connected ──────────────────────────────────────────────────
if not get_engine():
    st.warning("⚠️ No active database connection. Please connect first via **SQL Connect**.")
    st.stop()

meta = get_conn_meta()
st.info(
    f"Connected to **{meta['database']}** ({meta.get('dialect_label','DB')}) — read-only"
)

# ── Gemini setup ──────────────────────────────────────────────────────────────
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Gemini API setup failed: {e}")
    st.stop()

# ── Build schema context (cached per session until reconnect) ─────────────────
if "nl_schema_context" not in st.session_state:
    with st.spinner("Reading database schema…"):
        st.session_state["nl_schema_context"] = build_schema_context()

schema_ctx = st.session_state["nl_schema_context"]

with st.expander("📋 Schema sent to Gemini (click to inspect)"):
    st.code(schema_ctx, language="text")

if st.button("🔄 Refresh Schema"):
    st.session_state.pop("nl_schema_context", None)
    st.rerun()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are an expert SQL query writer.
You will be given a database schema and a plain-English question.
Your job is to write a single, correct, read-only SQL SELECT query that answers the question.

Rules:
1. Output ONLY the raw SQL query — no explanation, no markdown, no backticks.
2. Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, or any DDL/DML.
3. Use only the tables and columns that exist in the schema below.
4. Always alias aggregations for clarity (e.g. COUNT(*) AS total_count).
5. Do not add a LIMIT clause — the application adds it automatically.
6. Use the correct dialect for: {meta.get('dialect_label', 'SQL')}

Schema:
{schema_ctx}
""".strip()

# ── History helpers ───────────────────────────────────────────────────────────
_NL_HISTORY_KEY = "nl_query_history"
_NL_HISTORY_MAX = 10


def _push_nl_history(question: str, sql: str) -> None:
    history: list = st.session_state.get(_NL_HISTORY_KEY, [])
    entry = {"question": question.strip(), "sql": sql.strip()}
    history = [h for h in history if h["question"] != entry["question"]]
    history.insert(0, entry)
    st.session_state[_NL_HISTORY_KEY] = history[:_NL_HISTORY_MAX]


# ── NL input ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💬 Ask a Question")

question = st.text_area(
    "Your question",
    placeholder="e.g. What are the top 10 customers by total order value?",
    height=100,
    key="nl_question_input",
)

col_ask, col_load, _ = st.columns([1, 1, 4])
ask_clicked  = col_ask.button("🧠 Generate SQL")
load_clicked = col_load.button("📥 Load into AnalyticaOS")

# ── Generate SQL ──────────────────────────────────────────────────────────────
if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Gemini is writing your query…"):
            try:
                response = model.generate_content(
                    f"{SYSTEM_PROMPT}\n\nQuestion: {question.strip()}"
                )
                generated_sql = response.text.strip()

                # Strip accidental markdown fences if Gemini adds them
                if generated_sql.startswith("```"):
                    lines = generated_sql.splitlines()
                    generated_sql = "\n".join(
                        l for l in lines
                        if not l.strip().startswith("```")
                    ).strip()

                st.session_state["nl_generated_sql"] = generated_sql
                st.session_state["nl_last_question"] = question.strip()

            except Exception as e:
                st.error(f"Gemini error: {e}")

# ── Show generated SQL + execute ──────────────────────────────────────────────
if "nl_generated_sql" in st.session_state:
    st.divider()
    st.subheader("🔍 Generated SQL")

    edited_sql = st.text_area(
        "Review / edit before running",
        value=st.session_state["nl_generated_sql"],
        height=160,
        key="nl_sql_editor",
    )

    run_clicked = st.button("▶ Run Generated SQL")

    if run_clicked:
        with st.spinner("Executing…"):
            try:
                df_result = run_query(edited_sql)
                _push_nl_history(
                    st.session_state.get("nl_last_question", ""),
                    edited_sql,
                )
                st.session_state["nl_result_df"] = df_result
                st.success(
                    f"✅ {len(df_result):,} rows × "
                    f"{len(df_result.columns)} columns returned."
                )
            except Exception as e:
                st.error(f"Query error: {e}")

    if "nl_result_df" in st.session_state:
        df_preview = st.session_state["nl_result_df"]
        st.dataframe(
            df_preview.head(500).astype(str),
            use_container_width=True,
        )
        st.caption(
            f"Preview: first 500 rows. "
            f"Full result: {len(df_preview):,} rows."
        )

if load_clicked:
    if "nl_result_df" not in st.session_state:
        st.warning("Generate and run a query first before loading.")
    else:
        st.session_state["df"] = st.session_state["nl_result_df"].copy()
        st.session_state["data_source"] = (
            f"NL-to-SQL ({meta.get('dialect_label','DB')})"
        )
        st.success(
            "✅ Data loaded into AnalyticaOS. "
            "All downstream pages (EDA, ML, Forecast…) will now use this dataset."
        )

# ── NL Query history ──────────────────────────────────────────────────────────
nl_history: list = st.session_state.get(_NL_HISTORY_KEY, [])
if nl_history:
    st.divider()
    st.subheader("📜 Question History (last 10)")
    for i, entry in enumerate(nl_history):
        with st.expander(f"{i+1}. {entry['question'][:80]}"):
            st.code(entry["sql"], language="sql")
            if st.button("↩ Reload this query", key=f"nl_reload_{i}"):
                st.session_state["nl_generated_sql"] = entry["sql"]
                st.session_state["nl_last_question"] = entry["question"]
                st.rerun()