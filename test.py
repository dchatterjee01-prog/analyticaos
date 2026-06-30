"""
pre_push_check.py
Pre-push validation suite for AnalyticaOS.

Run this from the project root before pushing to GitHub or syncing to the
Hugging Face Space:

    conda activate analyticaos
    cd C:\\analyticaos
    python pre_push_check.py

What it checks:
  1. Every .py file under pages/, connections/, agents/, sql_agent/, config/
     compiles without a SyntaxError.
  2. The repo contains no known-dead patterns: gemini-2.0-flash, the old
     google.generativeai SDK (GenerativeModel / genai.configure), MySQL-only
     backtick row-count quoting in db_engine.py's fallback branch, etc.
  3. Functional smoke tests against a REAL temporary SQLite database:
     build_engine(), list_tables(), describe_table(), table_row_count(),
     run_query(), build_schema_context() — this is what actually catches
     bugs like the backtick row-count crash, not just "does it import."
  4. sql_guard.check_sql() correctly allows SELECT/WITH and blocks
     INSERT/DROP/etc.
  5. sql_builders / sql_query_builder generate syntactically sane SQL
     strings for window functions, CTEs, and the visual query builder.

Exit code is 0 if everything passes, 1 if anything fails — safe to wire
into a pre-commit hook or a CI step.

NOTE: this script imports your real connections/*.py modules, which import
streamlit. Running outside `streamlit run` means st.session_state falls
back to a plain dict-like object — you may see a harmless
"missing ScriptRunContext" warning printed to the console. That is
expected and does not affect the test results below.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")  # suppress Streamlit ScriptRunContext noise

PASS = []
FAIL = []


def ok(label: str):
    PASS.append(label)
    print(f"  ✅ {label}")


def bad(label: str, detail: str = ""):
    FAIL.append((label, detail))
    print(f"  ❌ {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"       {line}")


def section(title: str):
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


# ── Section 1: Syntax check every .py file in scope ─────────────────────────
def check_syntax():
    section("1. Syntax check (py_compile)")
    import py_compile

    scan_dirs = ["pages", "connections", "agents", "sql_agent", "config"]
    py_files = []
    for d in scan_dirs:
        full = ROOT / d
        if full.exists():
            py_files.extend(full.rglob("*.py"))

    # also check app.py at the root if present
    if (ROOT / "app.py").exists():
        py_files.append(ROOT / "app.py")

    if not py_files:
        bad(
            "No Python files found to check",
            f"Looked under: {', '.join(scan_dirs)} relative to {ROOT}\n"
            "Run this script from the project root (C:\\analyticaos).",
        )
        return

    for f in sorted(py_files):
        rel = f.relative_to(ROOT)
        try:
            py_compile.compile(str(f), doraise=True)
            ok(f"Syntax OK: {rel}")
        except py_compile.PyCompileError as e:
            bad(f"Syntax error: {rel}", str(e))


# ── Section 2: Grep for known-dead patterns ─────────────────────────────────
FORBIDDEN_PATTERNS = {
    r"gemini-2\.0-flash": (
        "Dead Gemini model (shut down June 1, 2026). "
        "Replace with gemini-2.5-flash."
    ),
    r"google\.generativeai": (
        "Old Gemini SDK import. Replace with `from google import genai` "
        "and the genai.Client(...) pattern."
    ),
    r"genai\.GenerativeModel\(": (
        "Old Gemini SDK class. Replace with "
        "client = genai.Client(api_key=...); "
        "client.models.generate_content(model=..., contents=...)."
    ),
    r"genai\.configure\(": (
        "Old Gemini SDK setup call. The new google.genai package has no "
        "configure() — use genai.Client(api_key=...) instead."
    ),
}

# Cosmetic/deprecation-level — reported but does not fail the run.
WARN_ONLY_PATTERNS = {
    r"use_container_width\s*=\s*True": (
        "Deprecated Streamlit param — prefer width='stretch'. "
        "Known existing drift, not a hard failure."
    ),
    r"\.style\.applymap\(": (
        "Deprecated pandas Styler method — prefer .style.map()."
    ),
    r"inplace\s*=\s*True": (
        "Avoid inplace=True on DataFrame ops per project convention — "
        "use df[col] = df[col].fillna(...) style instead."
    ),
}


def check_forbidden_patterns():
    section("2. Grep for known-dead / deprecated patterns")
    scan_dirs = ["pages", "connections", "agents", "sql_agent", "config"]
    py_files = []
    for d in scan_dirs:
        full = ROOT / d
        if full.exists():
            py_files.extend(full.rglob("*.py"))
    if (ROOT / "app.py").exists():
        py_files.append(ROOT / "app.py")

    any_forbidden_hit = False

    for f in sorted(py_files):
        rel = f.relative_to(ROOT)
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            bad(f"Could not read {rel}", str(e))
            continue

        for pattern, reason in FORBIDDEN_PATTERNS.items():
            matches = list(re.finditer(pattern, text))
            if matches:
                any_forbidden_hit = True
                lines = [
                    str(text.count("\n", 0, m.start()) + 1) for m in matches
                ]
                bad(
                    f"Forbidden pattern in {rel} (line(s) {', '.join(lines)})",
                    f"Pattern: {pattern}\nReason: {reason}",
                )

        for pattern, reason in WARN_ONLY_PATTERNS.items():
            matches = list(re.finditer(pattern, text))
            if matches:
                lines = [
                    str(text.count("\n", 0, m.start()) + 1) for m in matches
                ]
                print(
                    f"  ⚠️  {rel} (line(s) {', '.join(lines)}): {reason}"
                )

    if not any_forbidden_hit:
        ok("No dead Gemini SDK/model patterns found anywhere in scope")


# ── Section 3: Functional smoke test — real temp SQLite database ───────────
def check_db_engine_functional():
    section("3. Functional smoke test — connections.db_engine against a real SQLite DB")

    try:
        import pandas as pd
        from connections import db_engine
    except Exception as e:
        bad("Could not import connections.db_engine", traceback.format_exc())
        return

    tmp_path = os.path.join(tempfile.gettempdir(), "analyticaos_pretest.db")
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    try:
        db_engine.build_engine(
            host="localhost", port=0, database=tmp_path,
            username="", password="", dialect="sqlite",
        )
        ok("build_engine(dialect='sqlite') succeeded")
    except Exception:
        bad("build_engine(dialect='sqlite') failed", traceback.format_exc())
        return

    try:
        engine = db_engine.get_engine()
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "amount": [10.5, 20.0, 30.25, 40.0, 50.75],
        })
        df.to_sql("smoke_test_table", engine, if_exists="replace", index=False)
        ok("df.to_sql() created a table in the fresh SQLite DB")
    except Exception:
        bad("df.to_sql() failed", traceback.format_exc())
        return

    try:
        tables = db_engine.list_tables()
        assert "smoke_test_table" in tables, f"Expected table not found: {tables}"
        ok(f"list_tables() returned the new table: {tables}")
    except Exception:
        bad("list_tables() failed or table missing", traceback.format_exc())

    try:
        desc = db_engine.describe_table("smoke_test_table")
        assert not desc.empty, "describe_table() returned an empty frame"
        assert "id" in desc["name"].tolist(), "Expected column 'id' not found"
        ok(f"describe_table() returned {len(desc)} column(s) correctly")
    except Exception:
        bad("describe_table() failed", traceback.format_exc())

    try:
        # This is the exact call path that used to crash on the backtick bug.
        count = db_engine.table_row_count("smoke_test_table")
        assert count == 5, f"Expected 5 rows, got {count}"
        ok(f"table_row_count() returned correct count ({count}) — backtick fix verified")
    except Exception:
        bad(
            "table_row_count() failed — the SQLite quoting bug may have regressed",
            traceback.format_exc(),
        )

    try:
        result_df = db_engine.run_query("SELECT * FROM smoke_test_table WHERE amount > 20")
        assert len(result_df) == 3, f"Expected 3 rows, got {len(result_df)}"
        ok("run_query() executed a real SELECT and returned correct rows")
    except Exception:
        bad("run_query() failed", traceback.format_exc())

    try:
        ctx = db_engine.build_schema_context()
        assert "smoke_test_table" in ctx, "Schema context missing table name"
        ok("build_schema_context() produced a valid schema string for Gemini prompts")
    except Exception:
        bad("build_schema_context() failed", traceback.format_exc())

    # Negative test: writes must be rejected by run_query()'s SELECT/WITH guard
    try:
        try:
            db_engine.run_query("DELETE FROM smoke_test_table")
            bad("run_query() did NOT reject a DELETE statement", "This is a safety regression.")
        except ValueError:
            ok("run_query() correctly rejected a non-SELECT statement")
    except Exception:
        bad("Unexpected error during DELETE-rejection test", traceback.format_exc())

    # Cleanup
    try:
        db_engine.drop_engine()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        ok("Test database cleaned up")
    except Exception:
        bad("Cleanup failed (non-critical)", traceback.format_exc())


# ── Section 4: sql_guard functional test ────────────────────────────────────
def check_sql_guard():
    section("4. Functional smoke test — connections.sql_guard")
    try:
        from connections.sql_guard import check_sql
    except Exception:
        bad("Could not import connections.sql_guard", traceback.format_exc())
        return

    cases = [
        ("SELECT * FROM orders LIMIT 10", True),
        ("WITH t AS (SELECT 1) SELECT * FROM t", True),
        ("DELETE FROM orders", False),
        ("DROP TABLE orders", False),
        ("SELECT 1; DROP TABLE orders;", False),
    ]
    for sql, expected_safe in cases:
        try:
            result = check_sql(sql)
            if result.safe == expected_safe:
                ok(f"check_sql() correct for: {sql[:40]!r}")
            else:
                bad(
                    f"check_sql() WRONG for: {sql[:40]!r}",
                    f"Expected safe={expected_safe}, got safe={result.safe}",
                )
        except Exception:
            bad(f"check_sql() raised on: {sql[:40]!r}", traceback.format_exc())


# ── Section 5: SQL builders functional test ─────────────────────────────────
def check_sql_builders():
    section("5. Functional smoke test — connections.sql_builders & sql_query_builder")

    try:
        from connections.sql_builders import (
            WindowFunctionSpec, build_window_sql, CTEBlock, CTESpec, build_cte_sql,
        )
    except Exception:
        bad("Could not import connections.sql_builders", traceback.format_exc())
        return

    try:
        spec = WindowFunctionSpec(
            table="orders", func="ROW_NUMBER",
            order_cols=[("order_date", "DESC")],
            extra_cols=["id", "customer_id"],
        )
        sql = build_window_sql(spec)
        assert "ROW_NUMBER" in sql and "OVER" in sql
        ok("build_window_sql() produced valid SQL")
    except Exception:
        bad("build_window_sql() failed", traceback.format_exc())

    try:
        block = CTEBlock(name="recent_orders", sql="SELECT * FROM orders WHERE 1=1")
        cte_spec = CTESpec(blocks=[block], final_select="SELECT * FROM recent_orders")
        sql = build_cte_sql(cte_spec)
        assert "WITH" in sql and "recent_orders" in sql
        ok("build_cte_sql() produced valid SQL")
    except Exception:
        bad("build_cte_sql() failed", traceback.format_exc())

    try:
        from connections.sql_query_builder import (
            QueryBuilderSpec, AggregationSpec, build_query_sql, validate_spec,
        )
        spec2 = QueryBuilderSpec(
            table="orders",
            aggregations=[AggregationSpec(func="SUM", column="total", alias="total_sum")],
            group_by_cols=["customer_id"],
            limit=100,
        )
        errors = validate_spec(spec2)
        assert not errors, f"Unexpected validation errors: {errors}"
        sql, params = build_query_sql(spec2)
        assert "SUM" in sql and "GROUP BY" in sql
        ok("sql_query_builder.build_query_sql() produced valid SQL")
    except Exception:
        bad("sql_query_builder functional test failed", traceback.format_exc())


# ── Section 6: Gemini-calling modules — verify model string only (no API call) ──
def check_gemini_modules_static():
    section("6. Static check — Gemini-calling modules reference gemini-2.5-flash")
    targets = [
        ("connections/warehouse_advisor.py", "warehouse_advisor.py"),
        ("connections/auto_sql_agent.py", "auto_sql_agent.py"),
        ("connections/explainer.py", "explainer.py"),
        ("pages/sql_nlquery.py", "sql_nlquery.py"),
        ("pages/report.py", "report.py"),
        ("sql_agent/nl_to_sql.py", "nl_to_sql.py"),
    ]
    for rel_path, label in targets:
        f = ROOT / rel_path
        if not f.exists():
            print(f"  ⚠️  {label} not found at {rel_path} — skipped")
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if "gemini-2.5-flash" in text:
            ok(f"{label} references gemini-2.5-flash")
        else:
            bad(
                f"{label} does NOT reference gemini-2.5-flash",
                "Either the model string is missing/wrong, or this file "
                "doesn't call Gemini directly (verify manually).",
            )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("AnalyticaOS Pre-Push Validation")
    print("=" * 70)

    check_syntax()
    check_forbidden_patterns()
    check_db_engine_functional()
    check_sql_guard()
    check_sql_builders()
    check_gemini_modules_static()

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)

    if FAIL:
        print("\nFailed checks:")
        for label, detail in FAIL:
            print(f"  ❌ {label}")
        print(
            "\n🚫 DO NOT PUSH — fix the failures above first.\n"
        )
        sys.exit(1)
    else:
        print("\n✅ All checks passed — safe to push to GitHub / sync to Hugging Face.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()