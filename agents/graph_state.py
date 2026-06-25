"""
agents/graph_state.py
Stage E — Shared LangGraph state schema.

This TypedDict is the single contract every graph node reads/writes.
It preserves the existing AgentResult shape (status/summary/findings/
recommendations/artifacts/next_agents) so downstream consumers —
pages/agents_ui.py and pages/executive.py — do not need to change
their parsing logic when we cut over from Orchestrator to a
LangGraph CompiledGraph.

CRITICAL: findings dicts MUST keep the "severity" key (low/medium/high).
This is relied upon by agents_ui.py's SEVERITY_COLORS lookup and by
executive.py's LocalExecutiveEngine health-score calculation.

Step 3.1 hotfix: added run_counts to guard against infinite dynamic
re-triggering via next_agents (see graph_builder.py MAX_AGENT_RUNS).

Step 3.3 hotfix: next_agents is now a PLAIN field (no reducer), so each
node's update overwrites it (last-write-wins) instead of accumulating
via operator.add. Accumulating caused stale next_agents values from
earlier nodes (e.g. DataQualityAgent's own next_agents) to persist and
be misread as live routing instructions after later nodes ran, causing
an unintended duplicate DataQualityAgent <-> InsightAgent cycle.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict
import operator

import pandas as pd


def _merge_dict(left: dict, right: dict) -> dict:
    """Reducer: later nodes' artifacts/results win on key collision."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _merge_counts(left: dict, right: dict) -> dict:
    """Reducer: sums run counts per agent name across updates."""
    merged = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = merged.get(k, 0) + v
    return merged


class AgentFinding(TypedDict):
    title: str
    detail: str
    severity: str  # "low" | "medium" | "high" — DO NOT RENAME


class AgentState(TypedDict):
    """
    Shared state threaded through every LangGraph node.

    df            : the working DataFrame (not reduced — single source,
                    nodes read it, only a designated cleaning node writes it)
    context       : original context dict (aq_roadmap, aq_profile, target_col)
    pipeline      : ordered list of agent names that have run so far
    results       : agent_name -> raw AgentResult-shaped dict (merged across nodes)
    all_findings  : list of AgentFinding, accumulated via operator.add
    all_recommendations : list[str], accumulated via operator.add
    overall_status: "ok" | "warning" | "error" — last writer wins (plain field)
    next_agents   : agent names queued for conditional branching — PLAIN field,
                    last-write-wins. Each node reports only its own routing
                    intent; it must NOT accumulate across the run, since a
                    stale value from an earlier node would otherwise be
                    misread as live routing instruction after later nodes run.
    run_counts    : agent_name -> number of times that node has executed,
                    summed across the graph traversal (operator-style merge
                    via _merge_counts). Used by graph_builder.py to cap
                    dynamic re-triggering and prevent infinite loops.
    """
    df: pd.DataFrame
    context: dict[str, Any]
    pipeline: Annotated[list[str], operator.add]
    results: Annotated[dict[str, dict], _merge_dict]
    all_findings: Annotated[list[AgentFinding], operator.add]
    all_recommendations: Annotated[list[str], operator.add]
    overall_status: str
    next_agents: list[str]
    run_counts: Annotated[dict[str, int], _merge_counts]


def initial_state(df: pd.DataFrame, context: dict | None = None) -> AgentState:
    """Factory used by the graph's entry point."""
    return AgentState(
        df=df,
        context=context or {},
        pipeline=[],
        results={},
        all_findings=[],
        all_recommendations=[],
        overall_status="ok",
        next_agents=[],
        run_counts={},
    )