"""
Stage I — Autonomous Analytics SQL Agent (backend).
Gemini auto-generates analytical SQL queries from the connected schema
(revenue drivers, top-N, trends, anomaly candidates), executes each
through db_engine.run_query() (Stage H governed path), and returns
results ready to push into session_state["df"] for the existing
InsightAgent / question roadmap pipeline.
No Streamlit imports — pure Python, fully testable in isolation.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from google import genai
import streamlit as st
from connections.db_engine import (
    build_schema_context, get_conn_meta, run_query,
)
from connections.sql_guard import check_sql


# ── Query category constants ──────────────────────────────────────────────────
QUERY_CATEGORIES = [
    "top_n",
    "revenue_driver",
    "time_trend",
    "anomaly_candidate",
    "distribution",
    "correlation_proxy",
]

CATEGORY_LABELS = {
    "top_n":              "Top-N Ranking",
    "revenue_driver":     "Revenue Driver",
    "time_trend":         "Time Trend",
    "anomaly_candidate":  "Anomaly Candidate",
    "distribution":       "Distribution",
    "correlation_proxy":  "Correlation Proxy",
}

CATEGORY_ICONS = {
    "top_n":             "🏆",
    "revenue_driver":    "💰",
    "time_trend":        "📈",
    "anomaly_candidate": "🚨",
    "distribution":      "📊",
    "correlation_proxy": "🔗",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AutoQuery:
    """One auto-generated analytical SQL query and its result."""
    category:    str
    title:       str
    rationale:   str
    sql:         str
    result_df:   object = None   # pd.DataFrame once executed
    row_count:   int    = 0
    executed:    bool   = False
    error:       str    = ""


@dataclass
class AutoSQLResult:
    """Full output of one autonomous analytics run."""
    queries:     list[AutoQuery] = field(default_factory=list)
    schema_used: str             = ""
    raw_response:str             = ""
    parse_error: str             = ""

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_executed(self) -> int:
        # Only counting queries that executed successfully without errors
        return sum(1 for q in self.queries if q.executed and not q.error)

    @property
    def n_failed(self) -> int:
        return sum(1 for q in self.queries if q.executed and bool(q.error))


# ── Prompt ────────────────────────────────────────────────────────────────────

_AUTO_SQL_SYSTEM = """
You are a senior analytics engineer performing autonomous exploratory analysis.
Given a database schema, generate a set of analytical SQL SELECT queries that
would uncover the most important business insights without any user input.

Respond with ONLY a valid JSON array — no prose, no markdown, no backticks.

Each element must have exactly these keys:
{
  "category": "<one of: top_n, revenue_driver, time_trend, anomaly_candidate, distribution, correlation_proxy>",
  "title": "<short human-readable title, e.g. 'Top 10 Products by Revenue'>",
  "rationale": "<one sentence explaining what business question this answers>",
  "sql": "<a single valid SELECT or WITH...SELECT statement>"
}

Rules:
- Generate between 5 and 8 queries total — prioritize variety across categories.
- Every query must be a single SELECT or WITH...SELECT — never INSERT/UPDATE/DELETE/DROP.
- Always include a LIMIT clause (max 1000 rows) on non-aggregate queries.
- Use only tables and columns that exist in the provided schema.
- For time_trend queries, look for DATE or DATETIME columns to GROUP BY period.
- For anomaly_candidate queries, look for statistical outliers using aggregations
  (e.g. values > 2x the average, NULL-heavy columns, zero-value records).
- For correlation_proxy queries, JOIN two numeric measures from different tables.
- Use the dialect specified. Use backtick-quoted identifiers for MySQL/SQLite.
- If the schema has no numeric columns, skip revenue_driver and correlation_proxy.
- If the schema has no date columns, skip time_trend.
- Never refuse. Always return a valid JSON array even for simple schemas.
""".strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_queries(raw: str) -> tuple[list[AutoQuery], str]:
    """
    Parse Gemini's JSON array into AutoQuery objects.
    Returns (queries, parse_error). parse_error is empty string on success.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        items = json.loads(cleaned)
        if not isinstance(items, list):
            return [], "Gemini response was not a JSON array."

        queries = []
        for item in items:
            cat = item.get("category", "top_n")
            if cat not in QUERY_CATEGORIES:
                cat = "top_n"
            queries.append(AutoQuery(
                category  = cat,
                title     = item.get("title", "Untitled"),
                rationale = item.get("rationale", ""),
                sql       = item.get("sql", "").strip(),
            ))
        return queries, ""

    except json.JSONDecodeError as e:
        return [], f"JSON parse error: {e}"


# ── Execution ─────────────────────────────────────────────────────────────────

def _execute_query(query: AutoQuery) -> AutoQuery:
    """
    Run one AutoQuery through the governed execution path.
    Mutates and returns the same object with result_df / error filled in.
    """
    if not query.sql:
        query.error   = "Empty SQL — skipped."
        query.executed = True
        return query

    guard = check_sql(query.sql)
    if not guard.safe:
        query.error   = f"Safety check failed: {'; '.join(guard.blocked_clauses)}"
        query.executed = True
        return query

    try:
        df           = run_query(query.sql, limit=1000)
        query.result_df = df
        query.row_count = len(df)
        query.executed  = True
    except Exception as e:
        query.error   = str(e)
        query.executed = True

    return query


# ── Main entry point ──────────────────────────────────────────────────────────

def run_auto_analysis(
    categories: list[str] | None = None,
) -> AutoSQLResult:
    """
    Full autonomous analytics run:
      1. Build schema context from active connection
      2. Ask Gemini to generate analytical queries
      3. Guard-check and execute each through run_query()
      4. Return AutoSQLResult with populated DataFrames

    categories: optional filter — if provided, only queries in those
                categories are executed (all are still generated).
    Raises ValueError if no database connected.
    Raises RuntimeError on Gemini API failure.
    """
    schema_ctx = build_schema_context()
    if not schema_ctx:
        raise ValueError(
            "No active database connection or no tables found. "
            "Connect via SQL Connect first."
        )

    meta    = get_conn_meta()
    dialect = meta.get("dialect_label", "MySQL")

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Gemini API setup failed: {e}")

    prompt = (
        f"{_AUTO_SQL_SYSTEM}\n\n"
        f"Dialect: {dialect}\n\n"
        f"Schema:\n{schema_ctx}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw      = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    queries, parse_error = _parse_queries(raw)

    result = AutoSQLResult(
        queries      = queries,
        schema_used  = schema_ctx,
        raw_response = raw,
        parse_error  = parse_error,
    )

    if parse_error:
        return result

    # Filter by requested categories if specified
    to_run = [
        q for q in result.queries
        if categories is None or q.category in categories
    ]

    for query in to_run:
        _execute_query(query)

    return result


def best_result_for_pipeline(result: AutoSQLResult):
    """
    Pick the single best DataFrame from an AutoSQLResult to push into
    session_state["df"] for the existing InsightAgent pipeline.
    Priority: revenue_driver > top_n > distribution > others.
    Returns (AutoQuery, pd.DataFrame) or (None, None) if nothing executed.
    """
    priority = [
        "revenue_driver", "top_n", "distribution",
        "time_trend", "correlation_proxy", "anomaly_candidate",
    ]
    executed = [q for q in result.queries if q.executed and not q.error
                and q.result_df is not None and len(q.result_df) > 0]

    if not executed:
        return None, None

    for cat in priority:
        for q in executed:
            if q.category == cat:
                return q, q.result_df

    return executed[0], executed[0].result_df