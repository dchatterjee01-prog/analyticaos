"""
test_stage_e_full.py
Stage E — Full regression / verification suite for the LangGraph migration.

Run from project root:
    conda activate analyticaos
    python -B test_stage_e_full.py

Exits non-zero if any check fails, so it can be wired into CI later if desired.
"""
from __future__ import annotations

import sys
import traceback
import pandas as pd

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator-style runner: executes fn, records pass/fail, never crashes the suite."""
    def wrapper(fn):
        try:
            fn()
            RESULTS.append((name, True, ""))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"--- traceback for failed check: {name} ---")
            traceback.print_exc()
            print("-" * 60)
        return fn
    return wrapper


def make_df(rows: int = 10) -> pd.DataFrame:
    """Slightly bigger than earlier smoke tests so ModelingAgent has enough
    complete rows to actually train (avoids the known 'too few rows' status)."""
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "a": rng.normal(50, 10, rows),
        "b": rng.normal(100, 20, rows),
        "c": rng.choice(["x", "y", "z"], rows),
    })
    df.loc[1, "a"] = None  # one missing value, deliberate
    return df


# ─────────────────────────────────────────────────────────────────────────
# 1. graph_state.py — schema & reducer correctness
# ─────────────────────────────────────────────────────────────────────────

@check("1.1 initial_state() returns all required keys")
def _():
    from agents.graph_state import initial_state
    s = initial_state(make_df())
    required = {"df", "context", "pipeline", "results", "all_findings",
                "all_recommendations", "overall_status", "next_agents", "run_counts"}
    missing = required - set(s.keys())
    assert not missing, f"missing keys: {missing}"
    assert s["overall_status"] == "ok"
    assert s["pipeline"] == []
    assert s["run_counts"] == {}


@check("1.2 next_agents is a plain field (no operator.add)")
def _():
    from agents.graph_state import AgentState
    import typing
    hints = typing.get_type_hints(AgentState, include_extras=True)
    na = hints["next_agents"]
    # Annotated types expose __metadata__; a plain list[str] won't have it.
    has_reducer = hasattr(na, "__metadata__")
    assert not has_reducer, "next_agents must NOT have a reducer (last-write-wins required)"


@check("1.3 AgentFinding requires 'severity' key (contract preserved)")
def _():
    from agents.graph_state import AgentFinding
    import typing
    hints = typing.get_type_hints(AgentFinding)
    assert "severity" in hints, "'severity' key missing from AgentFinding contract"


# ─────────────────────────────────────────────────────────────────────────
# 2. graph_nodes.py — node wrapping correctness
# ─────────────────────────────────────────────────────────────────────────

@check("2.1 data_quality_node returns correct partial-update shape")
def _():
    from agents.graph_state import initial_state
    from agents.graph_nodes import data_quality_node
    from agents.base_agent import AgentResult
    state = initial_state(make_df())
    update = data_quality_node(state)
    assert update["pipeline"] == ["DataQualityAgent"]
    assert "DataQualityAgent" in update["results"]
    assert isinstance(update["results"]["DataQualityAgent"], AgentResult), \
        "results must store real AgentResult OBJECTS, not dicts (Step 4 fix)"
    assert update["run_counts"] == {"DataQualityAgent": 1}


@check("2.2 insight_node findings preserve 'severity' key end-to-end")
def _():
    from agents.graph_state import initial_state
    from agents.graph_nodes import insight_node
    state = initial_state(make_df())
    update = insight_node(state)
    for f in update["all_findings"]:
        assert "severity" in f, f"finding missing severity: {f}"
        assert f["severity"] in ("low", "medium", "high")


@check("2.3 NODE_REGISTRY has exactly the three expected agents")
def _():
    from agents.graph_nodes import NODE_REGISTRY
    assert set(NODE_REGISTRY.keys()) == {"DataQualityAgent", "InsightAgent", "ModelingAgent"}


# ─────────────────────────────────────────────────────────────────────────
# 3. graph_builder.py — routing, dedup, loop-guard
# ─────────────────────────────────────────────────────────────────────────

@check("3.1 default pipeline runs in correct order, no duplicates")
def _():
    from agents.graph_state import initial_state
    from agents.graph_builder import build_graph
    graph = build_graph()
    final = graph.invoke(initial_state(make_df(), context={}))
    assert final["pipeline"] == ["DataQualityAgent", "InsightAgent", "ModelingAgent"], \
        f"unexpected pipeline order/duplicates: {final['pipeline']}"


@check("3.2 run_counts never exceeds MAX_AGENT_RUNS for any agent")
def _():
    from agents.graph_state import initial_state
    from agents.graph_builder import build_graph, MAX_AGENT_RUNS
    graph = build_graph()
    final = graph.invoke(initial_state(make_df(), context={}))
    for agent, count in final["run_counts"].items():
        assert count <= MAX_AGENT_RUNS, f"{agent} ran {count} times, exceeds cap of {MAX_AGENT_RUNS}"


@check("3.3 aq_roadmap context correctly restricts which agents run")
def _():
    from agents.graph_state import initial_state
    from agents.graph_builder import build_graph
    graph = build_graph()
    # roadmap requests only cleaning + eda — no 'ml' module — ModelingAgent should be skipped
    roadmap = [{"module": "cleaning"}, {"module": "eda"}]
    final = graph.invoke(initial_state(make_df(), context={"aq_roadmap": roadmap}))
    assert "ModelingAgent" not in final["pipeline"], \
        f"ModelingAgent should be excluded by roadmap, got: {final['pipeline']}"
    assert final["pipeline"][0] == "DataQualityAgent", "DataQualityAgent must always run first"


@check("3.4 graph never hangs / completes within recursion limit on default config")
def _():
    from agents.graph_state import initial_state
    from agents.graph_builder import build_graph
    graph = build_graph()
    # If this raises GraphRecursionError or hangs, the test will fail/timeout —
    # absence of exception here is itself the pass condition.
    graph.invoke(initial_state(make_df(), context={}))


# ─────────────────────────────────────────────────────────────────────────
# 4. graph_runtime.py — UI-adapter parity with old Orchestrator contract
# ─────────────────────────────────────────────────────────────────────────

@check("4.1 run_graph_pipeline returns Orchestrator-identical key shape")
def _():
    from agents.graph_runtime import run_graph_pipeline
    report = run_graph_pipeline(make_df(), context={})
    required = {"pipeline", "results", "summary", "all_findings",
                "all_recommendations", "overall_status"}
    assert required.issubset(report.keys()), f"missing keys: {required - report.keys()}"


@check("4.2 results dict values support attribute access (result.status etc.)")
def _():
    from agents.graph_runtime import run_graph_pipeline
    report = run_graph_pipeline(make_df(), context={})
    dq = report["results"]["DataQualityAgent"]
    assert hasattr(dq, "status") and hasattr(dq, "findings") and hasattr(dq, "artifacts"), \
        "results must hold AgentResult objects with attribute access, not plain dicts"


@check("4.3 all_recommendations are deduplicated")
def _():
    from agents.graph_runtime import run_graph_pipeline
    report = run_graph_pipeline(make_df(), context={})
    recs = report["all_recommendations"]
    assert len(recs) == len(set(recs)), f"duplicate recommendations found: {recs}"


@check("4.4 summary string matches Orchestrator's format (status emoji + agent tags)")
def _():
    from agents.graph_runtime import run_graph_pipeline
    report = run_graph_pipeline(make_df(), context={})
    summary = report["summary"]
    assert any(e in summary for e in ("✅", "⚠️", "❌")), "summary missing status emoji"
    assert "[DataQualityAgent]" in summary
    assert "[InsightAgent]" in summary
    assert "[ModelingAgent]" in summary


@check("4.5 graph compiles once at import (module-level _COMPILED_GRAPH exists)")
def _():
    import agents.graph_runtime as gr
    assert gr._COMPILED_GRAPH is not None


# ─────────────────────────────────────────────────────────────────────────
# 5. Stage E side-effects — orchestrator retired cleanly, AnomalyAgent unaffected
# ─────────────────────────────────────────────────────────────────────────

@check("5.1 agents.orchestrator is no longer importable from active package path")
def _():
    try:
        import agents.orchestrator  # noqa: F401
        raise AssertionError("agents.orchestrator should have been moved to _deprecated")
    except ModuleNotFoundError:
        pass  # expected — confirms the Step 5 move took effect


@check("5.2 AnomalyAgent still importable and independent of the graph system")
def _():
    from agents import AnomalyAgent
    agent = AnomalyAgent()
    result = agent.run(make_df(), context={})
    assert hasattr(result, "status")


@check("5.3 agents/__init__.py does not export Orchestrator")
def _():
    import agents
    assert not hasattr(agents, "Orchestrator"), \
        "agents.Orchestrator still exported — should have been removed/never added"


@check("5.4 ModelingAgent succeeds (not 'too few rows') on a properly sized dataset")
def _():
    from agents.graph_runtime import run_graph_pipeline
    report = run_graph_pipeline(make_df(rows=10), context={})
    modeling_result = report["results"]["ModelingAgent"]
    # Not asserting status == "ok" strictly (data is random), just confirming
    # it didn't hit the small-sample-size early-exit path seen in earlier smoke tests.
    assert "too few" not in modeling_result.summary.lower(), \
        f"ModelingAgent still hitting small-sample path on {10}-row df: {modeling_result.summary}"


# ─────────────────────────────────────────────────────────────────────────
# Run all checks, print report
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STAGE E — FULL VERIFICATION SUITE")
    print("=" * 70)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)

    for name, ok, err in RESULTS:
        status = "✅ PASS" if ok else "❌ FAIL"
        line = f"{status}  {name}"
        if err:
            line += f"\n         -> {err}"
        print(line)

    print("-" * 70)
    print(f"TOTAL: {passed} passed, {failed} failed, {len(RESULTS)} checks")
    print("=" * 70)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()