"""
agents/graph_runtime.py
Stage E — Step 4: UI compatibility adapter.

Exposes run_graph_pipeline(df, context) with the EXACT same return shape
as the old Orchestrator.run(df, context) — pipeline / results / summary /
all_findings / all_recommendations / overall_status — so agents_ui.py and
executive.py can switch from Orchestrator to the LangGraph-backed graph
with a single import + call-site change, no rendering logic touched.

results values are real AgentResult objects (see graph_nodes.py Step 4
change), so result.status / result.findings / result.artifacts attribute
access in agents_ui.py keeps working unmodified.
"""
from __future__ import annotations

import pandas as pd

from agents.graph_state import initial_state
from agents.graph_builder import build_graph

# Build once at import time — StateGraph.compile() is not free, and the
# compiled graph is stateless/reusable across requests.
_COMPILED_GRAPH = build_graph()

_STATUS_LABEL = {
    "ok":      "✅ All checks passed",
    "warning": "⚠️ Issues detected",
    "error":   "❌ Pipeline errors",
}


def _build_summary(results: dict, overall_status: str) -> str:
    """Identical format to Orchestrator._build_summary, for UI parity."""
    parts = [f"[{name}] {result.summary}" for name, result in results.items()]
    status_label = _STATUS_LABEL.get(overall_status, overall_status)
    return f"{status_label}. " + " | ".join(parts)


def run_graph_pipeline(df: pd.DataFrame, context: dict | None = None) -> dict:
    """
    Drop-in replacement for Orchestrator().run(df, context).

    Returns
    -------
    dict with keys: pipeline, results, summary, all_findings,
    all_recommendations, overall_status — same shape Orchestrator produced,
    so existing UI code (agents_ui.py, executive.py) needs no changes
    beyond the call site itself.
    """
    state = _COMPILED_GRAPH.invoke(initial_state(df, context=context or {}))

    # all_recommendations may contain duplicates if multiple agents suggest
    # the same thing (Orchestrator deduplicated; we preserve that behavior)
    seen, dedup_recs = set(), []
    for r in state["all_recommendations"]:
        if r not in seen:
            seen.add(r)
            dedup_recs.append(r)

    return {
        "pipeline":            state["pipeline"],
        "results":             state["results"],
        "summary":             _build_summary(state["results"], state["overall_status"]),
        "all_findings":        state["all_findings"],
        "all_recommendations": dedup_recs,
        "overall_status":      state["overall_status"],
    }