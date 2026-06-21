"""
nl_to_sql.py
Converts a plain-English question into a single safe SQL SELECT
statement using Gemini 2.5 Flash, reusing the exact JSON-response and
safety-validation pattern established in pages/nlpqa.py for pandas.

Only the SCHEMA SUMMARY (from sql_agent.connection.fetch_schema_summary)
is ever sent to Gemini — never row data — matching the same privacy
philosophy used for the pandas-based Ask Your Data feature.
"""
import json
import re
import textwrap
from google import genai

from sql_agent.connection import validate_select_only

GEMINI_MODEL = "gemini-2.5-flash"


def _build_prompt(schema_summary: str, question: str) -> str:
    return textwrap.dedent(f"""
    You are a MySQL query generator. You are given the schema of a
    database (NOT the actual data — only table names, columns, types,
    and row counts).

    {schema_summary}

    User question: "{question}"

    Write a SINGLE MySQL SELECT statement that answers this question.

    RULES:
    - Output ONLY a SELECT statement — no INSERT/UPDATE/DELETE/DROP/ALTER.
    - Do NOT use semicolons or chain multiple statements.
    - Use backtick-quoted identifiers if a column/table name needs it.
    - If the question requires joining tables, join on columns that
      plausibly share the same real-world meaning (e.g. both tables
      having a 'movie_name' column).
    - If the question cannot be answered from the given schema, respond
      with sql = "" and explain why in the explanation field.
    - Always include a LIMIT clause (default 100) unless the user's
      question clearly asks for an aggregate (COUNT, SUM, AVG, etc.)
      that returns a small number of rows.

    Respond with ONLY a JSON object in this exact format, no markdown
    fences, no explanation outside the JSON:
    {{"sql": "<the SELECT statement>", "explanation": "<one sentence in plain English>"}}
    """).strip()


def generate_sql(question: str, schema_summary: str, api_key: str) -> dict:
    """
    Calls Gemini to generate SQL for the given question, then validates
    it through the same _is_select_only() safety check used by
    run_readonly_query(). Returns a dict with either:
      {"sql": "...", "explanation": "..."}  on success, or
      {"error": "..."}                       on any failure/rejection.
    The caller (Step 3's UI) is responsible for SHOWING the SQL to the
    user before ever executing it — this function only generates and
    validates, it does not run anything against the database.
    """
    if not api_key:
        return {"error": "GEMINI_API_KEY not configured."}

    prompt = _build_prompt(schema_summary, question)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        raw_text = response.text.strip()
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text).strip()
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"error": "Gemini did not return valid JSON. Try rephrasing your question."}
    except Exception as e:
        return {"error": f"Gemini API error: {e}"}

    sql = parsed.get("sql", "").strip()
    explanation = parsed.get("explanation", "")

    if not sql:
        return {"error": explanation or "Gemini could not generate a query for this question."}

    is_safe, reason = validate_select_only(sql)
    if not is_safe:
        return {"error": f"Generated query rejected for safety: {reason}", "rejected_sql": sql}

    return {"sql": sql, "explanation": explanation}
