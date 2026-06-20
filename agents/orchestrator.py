"""
orchestrator.py
The Orchestrator reads aq_roadmap (from Phase 8) and sequences
agent calls automatically. Falls back to a default pipeline
if no roadmap is available.

Usage:
    from agents.orchestrator import Orchestrator
    orch = Orchestrator()
    report = orch.run(df, context={"aq_roadmap": roadmap_list})
"""
import pandas as pd
from agents.base_agent import AgentResult
from agents.data_quality_agent import DataQualityAgent
from agents.insight_agent import InsightAgent
from agents.modeling_agent import ModelingAgent


# Maps roadmap module names → agent classes
AGENT_REGISTRY = {
    "cleaning":   DataQualityAgent,
    "data_quality": DataQualityAgent,
    "eda":        InsightAgent,
    "insight":    InsightAgent,
    "stats":      InsightAgent,
    "ml":         ModelingAgent,
    "modeling":   ModelingAgent,
}

# Default pipeline when no roadmap is present
DEFAULT_PIPELINE = ["DataQualityAgent", "InsightAgent", "ModelingAgent"]

NAME_TO_CLASS = {
    "DataQualityAgent": DataQualityAgent,
    "InsightAgent":     InsightAgent,
    "ModelingAgent":    ModelingAgent,
}


class Orchestrator:
    """
    Sequences agent calls and aggregates results into a single report.
    """

    def run(
        self,
        df: pd.DataFrame,
        context: dict | None = None,
        stop_on_error: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        df             : working DataFrame
        context        : shared context dict; may include:
                           aq_roadmap  — list of roadmap step dicts from Phase 8
                           aq_profile  — column profile dict from Phase 8
                           target_col  — override target column for ModelingAgent
        stop_on_error  : if True, halt pipeline on first agent error

        Returns
        -------
        dict with keys:
            pipeline   : list of agent names actually run
            results    : dict[agent_name → AgentResult]
            summary    : overall plain-language summary string
            all_findings        : merged findings list
            all_recommendations : deduplicated recommendations list
            overall_status      : "ok" | "warning" | "error"
        """
        context = context or {}
        pipeline = self._build_pipeline(context)

        results             = {}
        all_findings        = []
        all_recommendations = []
        overall_status      = "ok"

        for agent_name in pipeline:
            AgentClass = NAME_TO_CLASS.get(agent_name)
            if AgentClass is None:
                continue

            agent  = AgentClass()
            result: AgentResult = agent.run(df, context=context)
            results[agent_name] = result

            all_findings        += result.findings
            all_recommendations += result.recommendations

            if result.status == "error":
                overall_status = "error"
                if stop_on_error:
                    break
            elif result.status == "warning" and overall_status != "error":
                overall_status = "warning"

        # deduplicate recommendations while preserving order
        seen  = set()
        dedup = []
        for r in all_recommendations:
            if r not in seen:
                seen.add(r)
                dedup.append(r)

        summary = self._build_summary(results, overall_status)

        return {
            "pipeline":            pipeline,
            "results":             results,
            "summary":             summary,
            "all_findings":        all_findings,
            "all_recommendations": dedup,
            "overall_status":      overall_status,
        }

    # ── private helpers ────────────────────────────────────────────────────

    def _build_pipeline(self, context: dict) -> list[str]:
        """
        Convert aq_roadmap into an ordered agent list.
        Falls back to DEFAULT_PIPELINE if no roadmap.
        """
        roadmap = context.get("aq_roadmap", [])
        if not roadmap:
            return DEFAULT_PIPELINE.copy()

        seen     = set()
        pipeline = []

        # roadmap steps are dicts with a "module" key (e.g. "cleaning", "eda", "ml")
        for step in roadmap:
            module = step.get("module", "").lower()
            agent_class = AGENT_REGISTRY.get(module)
            if agent_class is None:
                continue
            name = agent_class.name  # class-level attribute
            # Use class name string for NAME_TO_CLASS lookup
            class_name = agent_class.__name__
            if class_name not in seen:
                seen.add(class_name)
                pipeline.append(class_name)

        # always ensure DataQualityAgent runs first
        if "DataQualityAgent" not in pipeline:
            pipeline.insert(0, "DataQualityAgent")

        return pipeline if pipeline else DEFAULT_PIPELINE.copy()

    def _build_summary(self, results: dict, overall_status: str) -> str:
        parts = []
        for name, result in results.items():
            parts.append(f"[{name}] {result.summary}")
        status_label = {
            "ok":      "✅ All checks passed",
            "warning": "⚠️ Issues detected",
            "error":   "❌ Pipeline errors",
        }.get(overall_status, overall_status)
        return f"{status_label}. " + " | ".join(parts)