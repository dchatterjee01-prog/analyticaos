"""
Stage I — Data Warehouse Schema Advisor.
Sends the connected database schema to Gemini and receives:
  - fact/dimension table recommendations
  - star schema design rationale
  - DDL to create the proposed structure
No Streamlit imports — pure Python, fully testable in isolation.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from google import genai
import streamlit as st
from connections.db_engine import build_schema_context, get_conn_meta


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class WarehouseAdvice:
    fact_tables:       list[dict]  = field(default_factory=list)
    dimension_tables:  list[dict]  = field(default_factory=list)
    schema_rationale:  str         = ""
    ddl_statements:    list[str]   = field(default_factory=list)
    warnings:          list[str]   = field(default_factory=list)
    raw_response:      str         = ""

    @property
    def n_fact(self) -> int:
        return len(self.fact_tables)

    @property
    def n_dim(self) -> int:
        return len(self.dimension_tables)

    @property
    def n_ddl(self) -> int:
        return len(self.ddl_statements)


# ── Prompt ────────────────────────────────────────────────────────────────────

_WAREHOUSE_SYSTEM = """
You are a senior data warehouse architect and analytics engineer.
Given a relational database schema, you must propose a star schema redesign
suitable for analytics and BI reporting.

Respond with ONLY a valid JSON object — no prose, no markdown, no backticks.

The JSON must have exactly these keys:
{
  "fact_tables": [
    {
      "name": "<proposed fact table name>",
      "source_tables": ["<original table(s) this is derived from>"],
      "measures": ["<numeric measure columns>"],
      "foreign_keys": ["<dimension table references>"],
      "rationale": "<why this is a fact table>"
    }
  ],
  "dimension_tables": [
    {
      "name": "<proposed dimension table name>",
      "source_tables": ["<original table(s)>"],
      "attributes": ["<descriptive attribute columns>"],
      "primary_key": "<surrogate or natural key>",
      "rationale": "<why this is a dimension table>"
    }
  ],
  "schema_rationale": "<2-3 sentence explanation of the overall star schema design>",
  "ddl_statements": [
    "<CREATE TABLE statement for dim_1>",
    "<CREATE TABLE statement for dim_2>",
    "<CREATE TABLE statement for fact_1>"
  ],
  "warnings": ["<any concerns about the existing schema, e.g. missing PKs, denormalization issues>"]
}

Rules:
- DDL must use the same SQL dialect specified in the prompt.
- Dimension tables must be created BEFORE fact tables in ddl_statements
  (foreign key dependency order).
- Use surrogate integer primary keys (id INTEGER PRIMARY KEY AUTO_INCREMENT
  for MySQL, id INTEGER PRIMARY KEY AUTOINCREMENT for SQLite).
- Prefix fact tables with fact_ and dimension tables with dim_.
- measures in fact tables should be numeric columns suitable for SUM/AVG/COUNT.
- If the schema is already well-structured for analytics, say so in
  schema_rationale and propose minimal changes.
- Never refuse. Always return valid JSON.
""".strip()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> WarehouseAdvice:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return WarehouseAdvice(
            fact_tables      = data.get("fact_tables", []),
            dimension_tables = data.get("dimension_tables", []),
            schema_rationale = data.get("schema_rationale", ""),
            ddl_statements   = data.get("ddl_statements", []),
            warnings         = data.get("warnings", []),
            raw_response     = raw,
        )
    except json.JSONDecodeError:
        return WarehouseAdvice(
            schema_rationale = cleaned,
            warnings         = ["Gemini response could not be parsed as JSON."],
            raw_response     = raw,
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_schema_for_warehouse() -> WarehouseAdvice:
    """
    Sends the connected database schema to Gemini and returns
    a WarehouseAdvice with star schema recommendations and DDL.
    Raises RuntimeError on Gemini API failure.
    Raises ValueError if no database is connected.
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
        client  = genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Gemini API setup failed: {e}")

    prompt = (
        f"{_WAREHOUSE_SYSTEM}\n\n"
        f"Dialect: {dialect}\n\n"
        f"Current Schema:\n{schema_ctx}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
        )
        raw = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    return _parse_response(raw)


def ddl_safe_for_display(ddl: str) -> bool:
    """
    Returns True if a DDL statement is CREATE TABLE only — safe to display.
    Filters out any accidental DROP/ALTER from Gemini.
    """
    upper = ddl.strip().upper()
    return upper.startswith("CREATE TABLE") and "DROP" not in upper