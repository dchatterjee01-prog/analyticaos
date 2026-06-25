"""
agents/graph_builder.py
Stage E — StateGraph construction.

Replaces Orchestrator._build_pipeline's roadmap-to-pipeline logic with
LangGraph's native graph + conditional edges. Preserves two rules from
the old Orchestrator:
  1. DataQualityAgent always runs first.
  2. aq_roadmap (if present) determines which of InsightAgent/ModelingAgent
     run; falls back to DEFAULT_PIPELINE if absent.

Dynamic routing: after each node, a conditional edge inspects
state["next_agents"] (populated by the wrapped agent's AgentResult) to
decide whether to branch to an agent not already in the static pipeline.

Step 3.1 hotfix: added MAX_AGENT_RUNS guard. A prior version of this
file allowed unbounded dynamic re-triggering via next_agents, which
caused an infinite DataQualityAgent <-> InsightAgent loop in testing
(observed as a hang inside insight_node / pandas describe()). Every
routing function now checks run_counts before allowing a node to fire
again, capping each agent at MAX_AGENT_RUNS executions per graph run.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agents.graph_state import AgentState
from agents.graph_nodes import (
    data_quality_node,
    insight_node,
    modeling_node,
    NODE_REGISTRY,
)

# Same roadmap-module -> node-name mapping as the old AGENT_REGISTRY in orchestrator.py
ROADMAP_TO_NODE = {
    "cleaning":     "DataQualityAgent",
    "data_quality": "DataQualityAgent",
    "eda":          "InsightAgent",
    "insight":      "InsightAgent",
    "stats":        "InsightAgent",
    "ml":           "ModelingAgent",
    "modeling":     "ModelingAgent",
}

DEFAULT_PIPELINE = ["DataQualityAgent", "InsightAgent", "ModelingAgent"]

# Each agent may run once in the static pipeline pass, plus at most one
# extra dynamic re-trigger via next_agents. Bounds worst-case execution
# at 2 x len(NODE_REGISTRY) node calls — finite no matter what an agent
# puts in next_agents.
MAX_AGENT_RUNS = 2


def _resolve_static_pipeline(context: dict) -> list[str]:
    """Mirrors Orchestrator._build_pipeline exactly, for parity during migration."""
    roadmap = context.get("aq_roadmap", [])
    if not roadmap:
        return DEFAULT_PIPELINE.copy()

    seen, pipeline = set(), []
    for step in roadmap:
        module = step.get("module", "").lower()
        node_name = ROADMAP_TO_NODE.get(module)
        if node_name and node_name not in seen:
            seen.add(node_name)
            pipeline.append(node_name)

    if "DataQualityAgent" not in pipeline:
        pipeline.insert(0, "DataQualityAgent")

    return pipeline if pipeline else DEFAULT_PIPELINE.copy()


def _can_run(state: AgentState, agent_name: str) -> bool:
    """True if agent_name has not yet hit MAX_AGENT_RUNS executions."""
    return state.get("run_counts", {}).get(agent_name, 0) < MAX_AGENT_RUNS


def _next_dynamic_candidate(state: AgentState, exclude: str) -> str | None:
    """
    First entry in state['next_agents'] that is a real node, isn't the
    agent that just ran, hasn't exhausted its run budget, AND is permitted
    by the roadmap (if one exists).
    """
    # Safely extract context, guaranteeing a dict even if state mapping is weird
    context = state.get("context") or {}
    roadmap = context.get("aq_roadmap")
    static_pipeline = _resolve_static_pipeline(context)

    for candidate in state.get("next_agents", []):
        if candidate in NODE_REGISTRY and candidate != exclude and _can_run(state, candidate):
            # Guard: If a strict roadmap is provided, ignore dynamic branches 
            # to agents that were explicitly excluded.
            if roadmap and candidate not in static_pipeline:
                continue
            return candidate
            
    return None


def _route_after_quality(state: AgentState) -> str:
    static_pipeline = _resolve_static_pipeline(state["context"])
    if "InsightAgent" in static_pipeline and _can_run(state, "InsightAgent"):
        return "InsightAgent"
    if "ModelingAgent" in static_pipeline and _can_run(state, "ModelingAgent"):
        return "ModelingAgent"
    return END


def _route_after_insight(state: AgentState) -> str:
    dynamic = _next_dynamic_candidate(state, exclude="InsightAgent")
    if dynamic:
        return dynamic
    static_pipeline = _resolve_static_pipeline(state["context"])
    if "ModelingAgent" in static_pipeline and _can_run(state, "ModelingAgent"):
        return "ModelingAgent"
    return END


def _route_after_modeling(state: AgentState) -> str:
    dynamic = _next_dynamic_candidate(state, exclude="ModelingAgent")
    if dynamic:
        return dynamic
    return END


def build_graph():
    """Compiles and returns the AnalyticaOS multi-agent StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("DataQualityAgent", data_quality_node)
    graph.add_node("InsightAgent", insight_node)
    graph.add_node("ModelingAgent", modeling_node)

    graph.set_entry_point("DataQualityAgent")

    graph.add_conditional_edges(
        "DataQualityAgent",
        _route_after_quality,
        {"InsightAgent": "InsightAgent", "ModelingAgent": "ModelingAgent", END: END},
    )
    graph.add_conditional_edges(
        "InsightAgent",
        _route_after_insight,
        {"ModelingAgent": "ModelingAgent", "DataQualityAgent": "DataQualityAgent", END: END},
    )
    graph.add_conditional_edges(
        "ModelingAgent",
        _route_after_modeling,
        {"InsightAgent": "InsightAgent", "DataQualityAgent": "DataQualityAgent", END: END},
    )

    return graph.compile()