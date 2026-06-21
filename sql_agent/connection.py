"""
connection.py
MySQL connection layer for the SQL Agent. Reads credentials from
.streamlit/secrets.toml (same pattern as GEMINI_API_KEY in nlpqa.py),
connects via mysql-connector-python, and exposes:
  - get_connection(): a live connection object
  - fetch_schema_summary(): a plain-text schema description (tables,
    columns, types, row counts) — sent to Gemini instead of raw data,
    same privacy pattern as nlpqa.py's _build_schema_summary()
  - run_readonly_query(): executes a SELECT-only query safely

SAFETY: run_readonly_query() refuses anything that isn't a SELECT
statement. Write operations (INSERT/UPDATE/DELETE/DROP/ALTER/etc.) are
explicitly out of scope for this step and must go through a separate,
deliberate opt-in mechanism if ever added later.
"""
import re
import mysql.connector
from mysql.connector import Error as MySQLError


# Statements that are always blocked, regardless of any future write-opt-in,
# because they aren't "data" writes — they're schema/destructive operations.
ALWAYS_BLOCKED = (
    "drop", "truncate", "alter", "grant", "revoke",
    "create user", "drop user", "shutdown",
)


def get_connection(secrets: dict):
    """
    Opens a MySQL connection using credentials from st.secrets (or any
    dict-like object with the same keys). Expected keys:
        MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    Optional: MYSQL_PORT (defaults to 3306)
    Raises MySQLError on failure — caller is responsible for catching it
    and showing a friendly message (see pages/sql_agent_ui.py, Step 3).
    """
    return mysql.connector.connect(
        host=secrets.get("MYSQL_HOST", "127.0.0.1"),
        port=int(secrets.get("MYSQL_PORT", 3306)),
        user=secrets["MYSQL_USER"],
        password=secrets["MYSQL_PASSWORD"],
        database=secrets["MYSQL_DATABASE"],
        connection_timeout=10,
    )


def fetch_schema_summary(conn) -> str:
    """
    Introspects every table in the connected database and returns a
    plain-text schema description: table names, columns, types, and
    row counts. This — NOT raw row data — is what gets sent to Gemini,
    mirroring the privacy pattern already established in nlpqa.py.
    """
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]

    lines = [f"Database has {len(tables)} table(s):\n"]

    for table in tables:
        cursor.execute(f"DESCRIBE `{table}`")
        columns = cursor.fetchall()
        # columns: (Field, Type, Null, Key, Default, Extra)

        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        row_count = cursor.fetchone()[0]

        lines.append(f"Table `{table}` ({row_count:,} rows):")
        for col in columns:
            field, col_type, nullable, key, default, extra = col
            key_note = f" [{key}]" if key else ""
            lines.append(f"  - {field} ({col_type}){key_note}")
        lines.append("")

    cursor.close()
    return "\n".join(lines)


def validate_select_only(sql: str) -> tuple[bool, str]:
    """Returns (is_safe, reason_if_not). Mirrors the safety philosophy
    in nlpqa.py's _validate_code(), adapted for SQL: only a single
    SELECT statement is allowed, no semicolon-chained statements, no
    destructive/DDL keywords anywhere in the text.

    Public (no underscore prefix) because nl_to_sql.py also needs this
    exact same check to validate Gemini-generated SQL before it's ever
    shown to the user or executed — duplicating this logic in two
    places would risk the two copies drifting out of sync over time."""
    stripped = sql.strip().rstrip(";").strip()

    if ";" in stripped:
        return False, "Multiple statements (semicolon-separated) are not allowed."

    if not re.match(r"^\s*select\b", stripped, re.IGNORECASE):
        return False, "Only SELECT statements are allowed."

    lowered = stripped.lower()
    for keyword in ALWAYS_BLOCKED:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return False, f"Query contains a blocked keyword: '{keyword}'"

    # Defense in depth: also block INSERT/UPDATE/DELETE even though the
    # leading-SELECT check above should already exclude them — covers
    # cases like a SELECT containing a subquery with a write keyword.
    for keyword in ("insert", "update", "delete"):
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return False, f"Query contains a write keyword: '{keyword}'"

    return True, ""


def run_readonly_query(conn, sql: str):
    """
    Executes a validated SELECT-only query and returns (columns, rows).
    Raises ValueError if the query fails the safety check — caller
    should show the rejection reason to the user, never execute blindly.
    """
    is_safe, reason = validate_select_only(sql)
    if not is_safe:
        raise ValueError(f"Query rejected for safety: {reason}")

    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return columns, rows
