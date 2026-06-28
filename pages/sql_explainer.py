"""
Stage N — SQL Query Explainer page.
Paste any SQL → Gemini explains it, flags bottlenecks,
suggests optimisations, and produces a rewritten query.
"""

import streamlit as st
from connections.db_engine import get_engine, get_conn_meta
from connections.explainer import explain_query, has_rewrite, ExplainerResult
from connections.query_store import save_query
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, TEXT_COLOR, ACCENT_COLOR,
    SUCCESS_BG, SUCCESS_COLOR, WARNING_BG, WARNING_COLOR,
)

# ── Stage G: tier-badge CSS, adapted from pages/stats.py's metric-card /
# verdict-badge pattern. Explanation tiers have no pass/fail verdict, so the
# pill shape from stats.py is repurposed as a colour-coded tier label
# (Executive / Analyst / Beginner) rather than a significance flag.
_TIER_COLORS = {
    "executive": (SUCCESS_BG, SUCCESS_COLOR),
    "analyst":   (WARNING_BG, WARNING_COLOR),
    "beginner":  (SURFACE_COLOR, ACCENT_COLOR),
}


def _inject_explainer_css():
    st.markdown(f"""
    <style>
      .tier-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {PRIMARY_COLOR}33;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
      }}
      .tier-badge {{
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
      }}
      .tier-body {{
        color: {TEXT_COLOR};
        font-size: 0.95rem;
        line-height: 1.5;
      }}
      .metric-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {PRIMARY_COLOR}33;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
      }}
      .metric-val {{
        font-size: 1.6rem;
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
    </style>
    """, unsafe_allow_html=True)


def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{value}</div>
      <div class="metric-lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _tier_block(label: str, icon: str, tier_key: str, body: str, empty_msg: str):
    bg, color = _TIER_COLORS[tier_key]
    badge = (
        f'<span class="tier-badge" style="background:{bg};color:{color};'
        f'border:1px solid {color};">{icon} {label}</span>'
    )
    content = body.strip() if body and body.strip() else f"<em>{empty_msg}</em>"
    st.markdown(
        f'<div class="tier-card">{badge}<div class="tier-body">{content}</div></div>',
        unsafe_allow_html=True,
    )


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SQL Explainer", page_icon="🔬", layout="wide")
_inject_explainer_css()
st.title("🔬 SQL Query Explainer")
st.caption(
    "Paste any SQL query — Gemini explains it at three levels "
    "(Executive, Analyst, Beginner), flags bottlenecks, and suggests an "
    "optimised rewrite."
)

# ── Dialect context (optional — works without a DB connection) ────────────────
meta           = get_conn_meta()
dialect_label  = meta.get("dialect_label", None)

if get_engine():
    st.info(
        f"Connected to **{meta['database']}** "
        f"({dialect_label}) — explanations will be dialect-specific."
    )
else:
    st.warning(
        "⚠️ No active database connection — "
        "explanations will use generic SQL. "
        "Connect via **SQL Connect** for dialect-specific advice."
    )
    # Let user pick dialect manually if not connected
    dialect_options = ["MySQL", "PostgreSQL", "SQLite", "SQL Server", "Generic SQL"]
    dialect_label   = st.selectbox(
        "Select SQL dialect for explanation",
        options=dialect_options,
        key="explainer_dialect_select",
    )

# ── SQL input ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("① Paste Your SQL Query")

# Pre-fill from other pages if available
prefill = (
    st.session_state.get("sql_last_query")
    or st.session_state.get("nl_generated_sql")
    or ""
)

sql_input = st.text_area(
    "SQL query to explain",
    value=st.session_state.get("explainer_sql_input", prefill),
    height=200,
    placeholder="SELECT o.customer_id, SUM(o.total) AS revenue\n"
                "FROM orders o\n"
                "JOIN customers c ON o.customer_id = c.id\n"
                "GROUP BY o.customer_id\n"
                "ORDER BY revenue DESC;",
    key="explainer_sql_input",
)

col_explain, col_clear, _ = st.columns([1, 1, 4])
explain_clicked = col_explain.button("🔬 Explain Query")
clear_clicked   = col_clear.button("🗑 Clear")

if clear_clicked:
    st.session_state.pop("explainer_result",    None)
    st.session_state.pop("explainer_sql_input", None)
    st.rerun()

# ── Run explainer ─────────────────────────────────────────────────────────────
if explain_clicked:
    if not sql_input.strip():
        st.warning("Please paste a SQL query first.")
    else:
        with st.spinner("Gemini is analysing your query…"):
            try:
                result: ExplainerResult = explain_query(
                    sql=sql_input,
                    dialect=dialect_label or "MySQL",
                )
                st.session_state["explainer_result"]    = result
                st.session_state["explainer_sql_cache"] = sql_input
            except Exception as e:
                st.error(f"Explainer error: {e}")

# ── Render result ─────────────────────────────────────────────────────────────
if "explainer_result" not in st.session_state:
    st.stop()

result: ExplainerResult = st.session_state["explainer_result"]
cached_sql: str         = st.session_state.get("explainer_sql_cache", "")

st.divider()

# ── Section 1: Three-Tier Explanation ────────────────────────────────────────
st.subheader("② Explanation — Three Levels")

m1, m2, m3 = st.columns(3)
_metric_card(m1, len(result.bottlenecks), "Bottlenecks Found")
_metric_card(m2, len(result.suggestions), "Suggestions")
_metric_card(m3, "Yes" if has_rewrite(result) else "No", "Rewrite Available")

st.markdown("<br>", unsafe_allow_html=True)

_tier_block(
    "Executive", "🎯", "executive",
    result.explanation_executive,
    "No executive summary returned.",
)
_tier_block(
    "Analyst", "📊", "analyst",
    result.explanation_analyst,
    "No analyst-level explanation returned.",
)
_tier_block(
    "Beginner", "🧑‍🎓", "beginner",
    result.explanation_beginner,
    "No beginner walkthrough returned.",
)

# ── Section 2: Bottlenecks ────────────────────────────────────────────────────
st.divider()
st.subheader("③ Bottlenecks Detected")

if not result.bottlenecks:
    st.success("✅ No bottlenecks detected — query looks clean.")
else:
    for i, b in enumerate(result.bottlenecks, 1):
        st.error(f"🔴 **{i}.** {b}")

# ── Section 3: Optimisation Suggestions ──────────────────────────────────────
st.divider()
st.subheader("④ Optimisation Suggestions")

if not result.suggestions:
    st.success("✅ No further suggestions.")
else:
    for i, s in enumerate(result.suggestions, 1):
        st.warning(f"💡 **{i}.** {s}")

# ── Section 4: Rewritten Query ────────────────────────────────────────────────
st.divider()
st.subheader("⑤ Rewritten Query")

if not has_rewrite(result):
    st.success("✅ Original query is already optimal — no rewrite needed.")
else:
    st.caption("Gemini's optimised version of your query:")

    col_orig, col_rewrite = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        st.code(cached_sql, language="sql")

    with col_rewrite:
        st.markdown("**Rewritten**")
        st.code(result.rewritten_sql, language="sql")

    # Rewrite reasons
    if result.rewrite_reasons:
        with st.expander("📝 What changed and why", expanded=True):
            for i, reason in enumerate(result.rewrite_reasons, 1):
                st.markdown(f"**{i}.** {reason}")

    st.divider()

    # Actions on rewritten query
    col_use, col_save, _ = st.columns([1, 1, 4])

    if col_use.button("📋 Use rewritten query in SQL Connect"):
        st.session_state["sql_last_query"] = result.rewritten_sql
        st.success("✅ Loaded into SQL Connect query box.")

    if col_save.button("💾 Save rewritten query"):
        st.session_state["explainer_show_save"] = True

    if st.session_state.get("explainer_show_save"):
        with st.form("explainer_save_form"):
            col_n, col_d = st.columns([2, 3])
            qname = col_n.text_input("Query name", key="explainer_save_name")
            qdesc = col_d.text_input(
                "Description (optional)", key="explainer_save_desc"
            )
            save_submitted = st.form_submit_button("Save")

        if save_submitted:
            if not qname.strip():
                st.error("Query name is required.")
            else:
                try:
                    save_query(
                        name        = qname.strip(),
                        sql         = result.rewritten_sql,
                        description = qdesc.strip(),
                        dialect     = (dialect_label or "mysql").lower(),
                        source_page = "SQL Explainer",
                    )
                    st.success(f"✅ Saved as **{qname.strip()}**.")
                    st.session_state.pop("explainer_show_save", None)
                except Exception as e:
                    st.error(f"Save failed: {e}")

# ── Debug expander ────────────────────────────────────────────────────────────
with st.expander("🐛 Raw Gemini response (debug)", expanded=False):
    st.code(result.raw_response or "(empty)", language="text")