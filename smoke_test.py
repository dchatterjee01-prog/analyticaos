# smoke_test.py
"""
Phase 12 smoke test — imports every page module and runs its entry point
with a dummy dataset loaded into session_state, using Streamlit's AppTest
framework (simulates a real Streamlit run context, so st.markdown/st.button/
etc. don't crash the way a direct show() call outside Streamlit would).

Run with:  python smoke_test.py
Exit code: 0 if all pages pass, 1 if any fail (CI-friendly).
"""
import sys
import pandas as pd
from streamlit.testing.v1 import AppTest

# module_path -> entry point function name (most use "show", upload uses "render")
PAGE_MODULES = {
    "pages.upload":     "render",
    "pages.cleaning":   "show",
    "pages.eda":        "show",
    "pages.viz":        "show",
    "pages.pivot":      "show",
    "pages.stats":      "show",
    "pages.ml":         "show",
    "pages.questions":  "show",
    "pages.agents_ui":  "show",
    "pages.executive":  "show",
    "pages.report":     "show",
    "pages.forecast":   "show",
    "pages.anomaly":    "show",
    "pages.nlpqa":      "show",
}


def _make_dummy_df():
    return pd.DataFrame({
        "revenue":    [100.0, 250.5, None, 180.2, 90.0, 310.7, 220.1, 150.0],
        "units_sold": [10, 25, 8, 18, 9, 31, 22, 15],
        "region":     ["North", "South", "North", "East", "West", "South", "North", "East"],
        "order_date": pd.date_range("2024-01-01", periods=8, freq="W"),
    })


def _run_page(module_path, func_name):
    import importlib
    import streamlit as st
    mod = importlib.import_module(module_path)
    entry_point = getattr(mod, func_name)
    entry_point()


def run_smoke_test():
    results = []
    for module_path, func_name in PAGE_MODULES.items():
        at = AppTest.from_function(
            _run_page,
            kwargs={"module_path": module_path, "func_name": func_name},
        )
        at.session_state["df"] = _make_dummy_df()
        at.session_state["filename"] = "smoke_test_data.csv"

        try:
            at.run(timeout=30)
            if at.exception:
                results.append((module_path, "FAIL", str(at.exception[0].value)))
            else:
                results.append((module_path, "PASS", ""))
        except Exception as e:
            results.append((module_path, "FAIL", f"{type(e).__name__}: {e}"))

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    failed = 0
    for module_path, status, detail in results:
        marker = "✅" if status == "PASS" else "❌"
        print(f"{marker} {status:<5} {module_path}")
        if status == "FAIL":
            failed += 1
            print(f"        → {detail}")
    print("=" * 60)
    print(f"{len(results) - failed}/{len(results)} pages passed.")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
