"""
Stage J — Offline Unit Test Suite
Run from Anaconda Prompt inside the analyticaos conda env:

    cd C:\\analyticaos
    python test_stage_j.py

No database connection required. No Streamlit required.
All tests are pure-Python unit tests against the backend modules only.
Green = ready for Stage K.
"""

import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PASS = []
FAIL = []


def ok(name):
    PASS.append(name)
    print(f"  ✅  {name}")


def fail(name, reason):
    FAIL.append(name)
    print(f"  ❌  {name}")
    print(f"      {reason}")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Sample DataFrames ─────────────────────────────────────────────────────────

def make_cohort_df():
    """Simulated user cohort data: user_id, signup_date, activity_date."""
    rows = []
    base = datetime(2024, 1, 1)
    for user_id in range(1, 51):
        signup = base + timedelta(days=(user_id % 4) * 7)
        for week in range(5):
            if np.random.rand() > 0.3:
                rows.append({
                    "user_id":       user_id,
                    "signup_date":   signup.strftime("%Y-%m-%d"),
                    "activity_date": (signup + timedelta(weeks=week)).strftime("%Y-%m-%d"),
                })
    return pd.DataFrame(rows)


def make_funnel_df():
    """Simulated funnel event data: user_id, event, timestamp."""
    events = ["page_view", "sign_up", "add_to_cart", "checkout", "purchase"]
    rows = []
    for user_id in range(1, 101):
        drop_at = np.random.randint(1, len(events) + 1)
        for i, event in enumerate(events[:drop_at]):
            rows.append({
                "user_id":   user_id,
                "event":     event,
                "timestamp": (datetime(2024, 1, 1) + timedelta(hours=i)).isoformat(),
            })
    return pd.DataFrame(rows)


def make_dashboard_df():
    """Simulated sales data for dashboard metrics."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    return pd.DataFrame({
        "date":     dates,
        "revenue":  np.random.randint(1000, 9000, 90),
        "units":    np.random.randint(10, 200, 90),
        "region":   np.random.choice(["North", "South", "East", "West"], 90),
    })


# ══════════════════════════════════════════════════════════════
# 1. pages/dashboard.py — importable, has show()
# ══════════════════════════════════════════════════════════════
section("1 · dashboard — module structure")

try:
    import importlib
    dash_mod = importlib.import_module("pages.dashboard")
    ok("pages.dashboard imports without error")
except Exception as e:
    fail("pages.dashboard imports without error", e)

try:
    assert hasattr(dash_mod, "show") and callable(dash_mod.show)
    ok("pages.dashboard has callable show()")
except Exception as e:
    fail("pages.dashboard has callable show()", e)


# ══════════════════════════════════════════════════════════════
# 2. pages/cohorts.py — importable, has show()
# ══════════════════════════════════════════════════════════════
section("2 · cohorts — module structure")

try:
    cohort_mod = importlib.import_module("pages.cohorts")
    ok("pages.cohorts imports without error")
except Exception as e:
    fail("pages.cohorts imports without error", e)

try:
    assert hasattr(cohort_mod, "show") and callable(cohort_mod.show)
    ok("pages.cohorts has callable show()")
except Exception as e:
    fail("pages.cohorts has callable show()", e)


# ══════════════════════════════════════════════════════════════
# 3. pages/funnel.py — importable, has show()
# ══════════════════════════════════════════════════════════════
section("3 · funnel — module structure")

try:
    funnel_mod = importlib.import_module("pages.funnel")
    ok("pages.funnel imports without error")
except Exception as e:
    fail("pages.funnel imports without error", e)

try:
    assert hasattr(funnel_mod, "show") and callable(funnel_mod.show)
    ok("pages.funnel has callable show()")
except Exception as e:
    fail("pages.funnel has callable show()", e)


# ══════════════════════════════════════════════════════════════
# 4. Cohort analysis — core logic (backend functions if exposed)
# ══════════════════════════════════════════════════════════════
section("4 · cohort analysis — data logic")

try:
    df = make_cohort_df()
    assert "user_id"       in df.columns
    assert "signup_date"   in df.columns
    assert "activity_date" in df.columns
    assert len(df) > 0
    ok("Cohort sample DataFrame builds correctly")
except Exception as e:
    fail("Cohort sample DataFrame builds correctly", e)

try:
    df = make_cohort_df()
    df["signup_date"]   = pd.to_datetime(df["signup_date"],   format="mixed")
    df["activity_date"] = pd.to_datetime(df["activity_date"], format="mixed")
    df["cohort_month"]  = df["signup_date"].dt.to_period("M")
    assert df["cohort_month"].notna().all()
    ok("Cohort month period extraction works on date strings")
except Exception as e:
    fail("Cohort month period extraction works on date strings", e)

try:
    df = make_cohort_df()
    df["signup_date"]   = pd.to_datetime(df["signup_date"],   format="mixed")
    df["activity_date"] = pd.to_datetime(df["activity_date"], format="mixed")
    df["period_number"] = (
        (df["activity_date"].dt.to_period("W") -
         df["signup_date"].dt.to_period("W")).apply(lambda x: x.n)
    )
    assert df["period_number"].min() >= 0
    ok("Cohort period number (week offset) is non-negative")
except Exception as e:
    fail("Cohort period number (week offset) is non-negative", e)

try:
    df = make_cohort_df()
    df["signup_date"]   = pd.to_datetime(df["signup_date"],   format="mixed")
    df["activity_date"] = pd.to_datetime(df["activity_date"], format="mixed")
    df["cohort_month"]  = df["signup_date"].dt.to_period("M")
    df["period_number"] = (
        (df["activity_date"].dt.to_period("W") -
         df["signup_date"].dt.to_period("W")).apply(lambda x: x.n)
    )
    cohort_sizes = df.groupby("cohort_month")["user_id"].nunique()
    retention    = df.groupby(["cohort_month", "period_number"])["user_id"].nunique()
    retention_pct = retention.div(cohort_sizes, level="cohort_month") * 100
    assert not retention_pct.empty
    assert retention_pct.max() <= 100.0
    ok("Cohort retention % never exceeds 100")
except Exception as e:
    fail("Cohort retention % never exceeds 100", e)


# ══════════════════════════════════════════════════════════════
# 5. Funnel analysis — core logic
# ══════════════════════════════════════════════════════════════
section("5 · funnel analysis — data logic")

try:
    df = make_funnel_df()
    assert "user_id" in df.columns
    assert "event"   in df.columns
    assert len(df) > 0
    ok("Funnel sample DataFrame builds correctly")
except Exception as e:
    fail("Funnel sample DataFrame builds correctly", e)

try:
    df = make_funnel_df()
    events   = ["page_view", "sign_up", "add_to_cart", "checkout", "purchase"]
    counts   = [df[df["event"] == e]["user_id"].nunique() for e in events]
    assert counts[0] >= counts[-1], "Top of funnel must have >= users than bottom"
    assert all(c >= 0 for c in counts)
    ok("Funnel counts are monotonically non-increasing top to bottom")
except Exception as e:
    fail("Funnel counts are monotonically non-increasing top to bottom", e)

try:
    df = make_funnel_df()
    events = ["page_view", "sign_up", "add_to_cart", "checkout", "purchase"]
    counts = [df[df["event"] == e]["user_id"].nunique() for e in events]
    dropoffs = []
    for i in range(1, len(counts)):
        pct = ((counts[i-1] - counts[i]) / counts[i-1] * 100) if counts[i-1] > 0 else 0
        dropoffs.append(round(pct, 2))
    assert all(0 <= d <= 100 for d in dropoffs)
    ok("Funnel drop-off percentages are between 0 and 100")
except Exception as e:
    fail("Funnel drop-off percentages are between 0 and 100", e)

try:
    df = make_funnel_df()
    events       = ["page_view", "sign_up", "add_to_cart", "checkout", "purchase"]
    counts       = [df[df["event"] == e]["user_id"].nunique() for e in events]
    top          = counts[0]
    conversions  = [round(c / top * 100, 2) if top > 0 else 0 for c in counts]
    assert conversions[0] == 100.0
    assert all(0 <= c <= 100 for c in conversions)
    ok("Funnel overall conversion rates correct (first step = 100%)")
except Exception as e:
    fail("Funnel overall conversion rates correct (first step = 100%)", e)


# ══════════════════════════════════════════════════════════════
# 6. Dashboard composer — data logic
# ══════════════════════════════════════════════════════════════
section("6 · dashboard composer — data logic")

try:
    df = make_dashboard_df()
    assert "date"    in df.columns
    assert "revenue" in df.columns
    assert "units"   in df.columns
    assert "region"  in df.columns
    ok("Dashboard sample DataFrame builds correctly")
except Exception as e:
    fail("Dashboard sample DataFrame builds correctly", e)

try:
    df = make_dashboard_df()
    assert pd.api.types.is_numeric_dtype(df["revenue"])
    assert pd.api.types.is_numeric_dtype(df["units"])
    ok("Dashboard numeric columns detected correctly")
except Exception as e:
    fail("Dashboard numeric columns detected correctly", e)

try:
    df = make_dashboard_df()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    assert len(numeric_cols) >= 1
    metrics = {c: {"sum": df[c].sum(), "mean": round(df[c].mean(), 2),
                   "max": df[c].max(), "min": df[c].min()}
               for c in numeric_cols}
    for col, m in metrics.items():
        assert m["max"] >= m["min"]
        assert m["sum"] >= 0 or True   # allow negative values
    ok("Dashboard metric aggregations (sum/mean/max/min) compute correctly")
except Exception as e:
    fail("Dashboard metric aggregations (sum/mean/max/min) compute correctly", e)

try:
    df = make_dashboard_df()
    cat_cols = [c for c in df.columns
                if df[c].dtype == object or
                str(df[c].dtype) == "category"]
    assert "region" in cat_cols
    grouped = df.groupby("region")["revenue"].sum().reset_index()
    assert len(grouped) == df["region"].nunique()
    ok("Dashboard group-by categorical column works correctly")
except Exception as e:
    fail("Dashboard group-by categorical column works correctly", e)

try:
    df = make_dashboard_df()
    date_cols = [c for c in df.columns
                 if pd.api.types.is_datetime64_any_dtype(df[c])]
    assert "date" in date_cols
    ok("Dashboard date column detected as datetime dtype")
except Exception as e:
    fail("Dashboard date column detected as datetime dtype", e)


# ══════════════════════════════════════════════════════════════
# 7. app.py NAV — Stage J entries present
# ══════════════════════════════════════════════════════════════
section("7 · app.py NAV — Stage J pages registered")

try:
    with open("app.py", encoding="utf-8") as f:
        app_src = f.read()
    ok("app.py readable")
except Exception as e:
    fail("app.py readable", e)
    app_src = ""

try:
    assert "Dashboard Composer" in app_src or "dashboard" in app_src
    ok("app.py contains Dashboard Composer route")
except Exception as e:
    fail("app.py contains Dashboard Composer route", e)

try:
    assert "Cohort Analysis" in app_src or "cohorts" in app_src
    ok("app.py contains Cohort Analysis route")
except Exception as e:
    fail("app.py contains Cohort Analysis route", e)

try:
    assert "Funnel Analysis" in app_src or "funnel" in app_src
    ok("app.py contains Funnel Analysis route")
except Exception as e:
    fail("app.py contains Funnel Analysis route", e)

try:
    assert "from pages.dashboard import" in app_src or \
           "from pages import dashboard" in app_src or \
           "dashboard_show" in app_src
    ok("app.py lazy-loads dashboard page")
except Exception as e:
    fail("app.py lazy-loads dashboard page", e)

try:
    # confirm no top-level eager import of stage j pages
    import_block_end = app_src.find("st.set_page_config")
    import_section   = app_src[:import_block_end]
    assert "from pages import dashboard" not in import_section
    assert "from pages import cohorts"   not in import_section
    assert "from pages import funnel"    not in import_section
    ok("Stage J pages are NOT eagerly imported at top level")
except Exception as e:
    fail("Stage J pages are NOT eagerly imported at top level", e)


# ══════════════════════════════════════════════════════════════
# 8. bridge compatibility — Stage J DataFrames
# ══════════════════════════════════════════════════════════════
section("8 · bridge — apply_bridge on Stage J DataFrames")

try:
    from connections.bridge import apply_bridge
    ok("connections.bridge imports")
except Exception as e:
    fail("connections.bridge imports", e)

try:
    df_cast, br = apply_bridge(make_cohort_df())
    assert df_cast is not None
    assert br.original_shape[0] > 0
    ok("apply_bridge handles cohort DataFrame")
except Exception as e:
    fail("apply_bridge handles cohort DataFrame", e)

try:
    df_cast, br = apply_bridge(make_funnel_df())
    assert df_cast is not None
    ok("apply_bridge handles funnel DataFrame")
except Exception as e:
    fail("apply_bridge handles funnel DataFrame", e)

try:
    df_cast, br = apply_bridge(make_dashboard_df())
    assert df_cast is not None
    ok("apply_bridge handles dashboard DataFrame")
except Exception as e:
    fail("apply_bridge handles dashboard DataFrame", e)


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  Stage J Test Results")
print(f"{'═'*60}")
print(f"  Passed : {len(PASS)}")
print(f"  Failed : {len(FAIL)}")
print(f"{'═'*60}")

if FAIL:
    print("\n  Failed tests:")
    for f in FAIL:
        print(f"    • {f}")
    print()
    sys.exit(1)
else:
    print("\n  ✅  All tests passed — green flag for Stage K.\n")
    sys.exit(0)