"""
Stage N + Stage G — SQL Query Explainer engine.
Sends SQL to Gemini and parses a structured response:
  - three-tier explanation (Executive / Analyst / Beginner)
  - bottleneck flags
  - optimisation suggestions
  - rewritten query
No Streamlit imports — pure Python, fully testable in isolation.
"""

from __future__ import annotations
import json
import re
from google import genai
import streamlit as st
from dataclasses import dataclass, field


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class ExplainerResult:
    """
    Structured output from Gemini SQL explanation.
    All fields have safe defaults — partial responses are tolerated.

    Stage G: 'explanation' is kept (Analyst-tier text, for backward
    compatibility with any other page reading this field) and supplemented
    with explicit tier fields so callers can choose what to render.
    """
    explanation_executive: str        = ""   # 1-2 sentences, business outcome, no jargon
    explanation_analyst:   str        = ""   # paragraph: joins/filters/aggregations logic
    explanation_beginner:  str        = ""   # line-by-line walkthrough of SQL concepts
    bottlenecks:           list[str]  = field(default_factory=list)
    suggestions:           list[str]  = field(default_factory=list)
    rewritten_sql:         str        = ""   # optimised query or empty if no changes needed
    rewrite_reasons:       list[str]  = field(default_factory=list)
    raw_response:          str        = ""   # full Gemini response for debugging

    @property
    def explanation(self) -> str:
        """
        Backward-compatible alias. Any existing caller reading
        result.explanation (pre-Stage-G) gets the Analyst tier, which is
        the closest match to the old single-tier output.
        """
        return self.explanation_analyst


# ── Prompt ────────────────────────────────────────────────────────────────────

_EXPLAINER_SYSTEM = """
You are an expert SQL analyst and performance engineer.
Given a SQL query, you must respond with ONLY a valid JSON object — no prose, 
no markdown, no backticks before or after.

The JSON must have exactly these keys:
{
  "explanation_executive": "<1-2 sentences, framed entirely around the business outcome/decision this query supports, zero technical jargon — no mention of joins, tables, SQL keywords, or syntax>",
  "explanation_analyst": "<one paragraph aimed at someone who knows SQL: explain the query's logic in terms of its joins, filters, grouping, and aggregations>",
  "explanation_beginner": "<a line-by-line walkthrough aimed at someone learning SQL: go through the query clause by clause, explaining each SQL concept as you introduce it, e.g. 'this CTE temporarily names a subquery so we can reference it later'>",
  "bottlenecks": ["<bottleneck 1>", "<bottleneck 2>"],
  "suggestions": ["<optimisation tip 1>", "<optimisation tip 2>"],
  "rewritten_sql": "<optimised SQL query, or empty string if no rewrite needed>",
  "rewrite_reasons": ["<reason for each change made in the rewrite>"]
}

Rules:
- explanation_executive: NEVER mention SQL, tables, columns, joins, or any
  technical term. Describe only what business question is being answered
  and what decision it could inform. Write for a non-technical executive
  reading a one-line summary.
- explanation_analyst: assumes SQL fluency. Be precise about the mechanics —
  which tables are joined and how, what's filtered, what's aggregated and by
  what grouping.
- explanation_beginner: assumes no prior SQL knowledge. Walk through the
  query roughly in execution order (or in written order if clearer),
  explaining each clause's purpose and naming the SQL concept involved
  (e.g. "a CTE", "an aggregate function", "an inner join") the first time
  it appears.
- bottlenecks: specific performance issues (missing index hints, SELECT *, 
  implicit type casts, non-sargable WHERE clauses, missing LIMIT, Cartesian joins).
  Empty list if none found.
- suggestions: concrete actionable improvements. Empty list if none.
- rewritten_sql: a corrected/optimised version of the query.
  If the original is already optimal, return an empty string.
- rewrite_reasons: one reason per change made. Empty list if rewritten_sql is empty.
- Never refuse. Always return valid JSON even for simple queries.
""".strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> ExplainerResult:
    """
    Parse Gemini's JSON response into an ExplainerResult.
    Tolerates markdown fences and extra whitespace.
    Falls back to raw_response on parse failure.
    Also tolerates the old single-key 'explanation' shape from a stale
    cached response or an older prompt version, mapping it to the
    Analyst tier so nothing silently goes blank.
    """
    cleaned = raw.strip()

    # Strip markdown fences if present
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        legacy_explanation = data.get("explanation", "")
        return ExplainerResult(
            explanation_executive = data.get("explanation_executive", ""),
            explanation_analyst   = data.get("explanation_analyst", legacy_explanation),
            explanation_beginner  = data.get("explanation_beginner", ""),
            bottlenecks           = data.get("bottlenecks", []),
            suggestions           = data.get("suggestions", []),
            rewritten_sql         = data.get("rewritten_sql", ""),
            rewrite_reasons       = data.get("rewrite_reasons", []),
            raw_response          = raw,
        )
    except json.JSONDecodeError:
        # Fallback: dump raw text into the Analyst tier so something
        # always renders, rather than failing silently.
        return ExplainerResult(
            explanation_analyst = cleaned,
            raw_response        = raw,
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def explain_query(
    sql: str,
    dialect: str = "MySQL",
) -> ExplainerResult:
    """
    Send sql to Gemini and return a structured ExplainerResult with all
    three explanation tiers populated.
    Raises ValueError on empty SQL.
    Raises RuntimeError on Gemini API failure.
    """
    if not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Gemini API setup failed: {e}")

    prompt = (
        f"{_EXPLAINER_SYSTEM}\n\n"
        f"Dialect: {dialect}\n\n"
        f"SQL Query:\n{sql.strip()}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw      = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    return _parse_response(raw)


def has_rewrite(result: ExplainerResult) -> bool:
    """True if Gemini produced a non-trivial rewritten query."""
    return bool(result.rewritten_sql.strip())