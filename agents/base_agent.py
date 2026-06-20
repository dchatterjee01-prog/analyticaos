"""
base_agent.py — Abstract base class for all AnalyticaOS agents.
Every agent returns a standardised AgentResult dict so the
Orchestrator can consume outputs uniformly.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Standardised output from any agent run."""
    agent_name: str
    status: str                        # "ok" | "warning" | "error"
    summary: str                       # one-sentence plain-language summary
    findings: list[dict]  = field(default_factory=list)   # list of {title, detail, severity}
    recommendations: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)  # named outputs (DataFrames, dicts…)
    next_agents: list[str] = field(default_factory=list)    # suggested follow-up agents


class BaseAgent(ABC):
    """All agents inherit from this."""

    name: str = "BaseAgent"

    @abstractmethod
    def run(self, df, context: dict | None = None) -> AgentResult:
        """
        Execute the agent.

        Parameters
        ----------
        df      : pd.DataFrame — the current working dataset
        context : dict         — shared session context (aq_profile, aq_roadmap, etc.)

        Returns
        -------
        AgentResult
        """
        ...