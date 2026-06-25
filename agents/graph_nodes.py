"""
agents/graph_nodes.py
Stage E — LangGraph node wrappers around the existing agents.

Each function has the LangGraph node signature: (AgentState) -> partial state dict.
The wrapped agents (DataQualityAgent, InsightAgent, ModelingAgent) are NOT modified —
this file only adapts their existing AgentResult output into AgentState updates.

CRITICAL: result.findings entries already carry "severity" (low/medium/high) from
the original agents — we pass them through unchanged, no remapping.

Step 3.1: each update reports "run_counts": {agent_name: 1} so graph_builder.py
can cap dynamic re-triggering via next_agents and prevent infinite loops.

Step 4: results now stores the real AgentResult OBJECT (not .__dict__), so
downstream consumers (pages/agents_ui.py, pages/executive.py) can keep using
attribute access (result.status, result.findings, result.artifacts) with zero
changes to their rendering logic.
"""
from __future__ import annotations

from agents.graph_state import AgentState
from agents.base_agent import AgentResult
from agents.data_quality_agent import DataQualityAgent
from agents.insight_agent import InsightAgent
from agents.modeling_agent import ModelingAgent


def _result_to_update(agent_name: str, result: AgentResult, state: AgentState) -> dict:
    """
    Shared conversion: AgentResult -> partial AgentState update.
    Reducers in graph_state.py handle merging (operator.add for lists,
    _merge_dict for the results dict, _merge_counts for run_counts) so
    we only need to return the NEW pieces this node contributes, not
    the full accumulated state.
    """
    new_status = state.get("overall_status", "ok")
    if result.status == "error":
        new_status = "error"
    elif result.status == "warning" and new_status != "error":
        new_status = "warning"

    return {
        "pipeline": [agent_name],
        "results": {agent_name: result},  # real AgentResult object, not __dict__
        "all_findings": list(result.findings),
        "all_recommendations": list(result.recommendations),
        "overall_status": new_status,
        "next_agents": list(getattr(result, "next_agents", []) or []),
        "run_counts": {agent_name: 1},
    }


def data_quality_node(state: AgentState) -> dict:
    """LangGraph node wrapping DataQualityAgent."""
    agent = DataQualityAgent()
    result: AgentResult = agent.run(state["df"], context=state["context"])
    return _result_to_update("DataQualityAgent", result, state)


def insight_node(state: AgentState) -> dict:
    """LangGraph node wrapping InsightAgent."""
    agent = InsightAgent()
    result: AgentResult = agent.run(state["df"], context=state["context"])
    return _result_to_update("InsightAgent", result, state)


def modeling_node(state: AgentState) -> dict:
    """LangGraph node wrapping ModelingAgent."""
    agent = ModelingAgent()
    result: AgentResult = agent.run(state["df"], context=state["context"])
    return _result_to_update("ModelingAgent", result, state)


# Name -> node function lookup, used when wiring StateGraph in graph_builder.py
NODE_REGISTRY = {
    "DataQualityAgent": data_quality_node,
    "InsightAgent":      insight_node,
    "ModelingAgent":     modeling_node,
}