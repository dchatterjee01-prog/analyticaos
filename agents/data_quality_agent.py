"""
data_quality_agent.py
Wraps the logic of Phase 2 (Cleaning Engine) into a callable agent.
Runs silently on the DataFrame and returns an AgentResult with:
  - missing value audit
  - duplicate row count
  - type inconsistency flags
  - overall quality score (0-100)
"""
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent, AgentResult


class DataQualityAgent(BaseAgent):
    name = "DataQualityAgent"

    # Thresholds
    MISSING_WARN  = 0.05   # 5 % missing → warning
    MISSING_HIGH  = 0.20   # 20 % missing → high severity
    DUP_WARN      = 0.01   # 1 % duplicates → warning

    def run(self, df: pd.DataFrame, context: dict | None = None) -> AgentResult:
        findings      = []
        recommendations = []
        artifacts     = {}

        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary="Empty DataFrame — cannot assess quality.",
            )

        # ── 1. Missing values ──────────────────────────────────────────────
        missing_counts = df.isnull().sum()
        missing_pct    = missing_counts / len(df)
        missing_df     = pd.DataFrame({
            "column":      missing_counts.index,
            "missing_n":   missing_counts.values,
            "missing_pct": (missing_pct.values * 100).round(2),
        }).query("missing_n > 0").sort_values("missing_pct", ascending=False).reset_index(drop=True)

        artifacts["missing_summary"] = missing_df

        for _, row in missing_df.iterrows():
            pct = row["missing_pct"] / 100
            sev = "high" if pct >= self.MISSING_HIGH else ("medium" if pct >= self.MISSING_WARN else "low")
            findings.append({
                "title":    f"Missing values in '{row['column']}'",
                "detail":   f"{row['missing_n']} rows ({row['missing_pct']}%) are null.",
                "severity": sev,
            })
            if sev in ("medium", "high"):
                recommendations.append(
                    f"Impute or drop '{row['column']}' — {row['missing_pct']}% missing."
                )

        # ── 2. Duplicate rows ──────────────────────────────────────────────
        dup_count = df.duplicated().sum()
        dup_pct   = dup_count / len(df)
        artifacts["duplicate_count"] = int(dup_count)

        if dup_count > 0:
            sev = "high" if dup_pct >= 0.05 else ("medium" if dup_pct >= self.DUP_WARN else "low")
            findings.append({
                "title":    "Duplicate rows detected",
                "detail":   f"{dup_count} duplicate rows ({dup_pct*100:.2f}%).",
                "severity": sev,
            })
            if sev in ("medium", "high"):
                recommendations.append("Remove duplicate rows before analysis.")

        # ── 3. Dtype inconsistencies — numeric stored as object ────────────
        suspicious = []
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(200)
            converted = pd.to_numeric(sample, errors="coerce")
            if converted.notna().mean() > 0.85:
                suspicious.append(col)

        artifacts["suspicious_type_cols"] = suspicious
        for col in suspicious:
            findings.append({
                "title":    f"Possible numeric stored as text: '{col}'",
                "detail":   "Over 85% of sampled values parse as numbers.",
                "severity": "medium",
            })
            recommendations.append(f"Convert '{col}' to numeric dtype.")

        # ── 4. Constant / near-constant columns ───────────────────────────
        for col in df.columns:
            n_unique = df[col].nunique(dropna=False)
            if n_unique == 1:
                findings.append({
                    "title":    f"Constant column: '{col}'",
                    "detail":   "Only one unique value — carries no information.",
                    "severity": "medium",
                })
                recommendations.append(f"Drop constant column '{col}'.")

        # ── 5. Quality score (simple heuristic) ───────────────────────────
        penalty = 0
        # missing penalty: up to 40 pts
        overall_missing_pct = df.isnull().mean().mean()
        penalty += min(40, overall_missing_pct * 200)
        # duplicate penalty: up to 20 pts
        penalty += min(20, dup_pct * 400)
        # type issues: 5 pts each, up to 20
        penalty += min(20, len(suspicious) * 5)
        # constant cols: 5 pts each, up to 20
        n_const = sum(1 for c in df.columns if df[c].nunique(dropna=False) == 1)
        penalty += min(20, n_const * 5)

        quality_score = max(0, round(100 - penalty, 1))
        artifacts["quality_score"] = quality_score

        # ── 6. Status & summary ───────────────────────────────────────────
        high_count = sum(1 for f in findings if f["severity"] == "high")
        if high_count > 0 or quality_score < 60:
            status = "warning"
        elif findings:
            status = "warning"
        else:
            status = "ok"

        summary = (
            f"Quality score {quality_score}/100. "
            f"{len(missing_df)} columns have missing values, "
            f"{dup_count} duplicate rows, "
            f"{len(suspicious)} type inconsistency issue(s)."
        )

        next_agents = []
        if quality_score < 80:
            next_agents.append("DataQualityAgent")   # re-run after cleaning
        next_agents.append("InsightAgent")

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary,
            findings=findings,
            recommendations=list(dict.fromkeys(recommendations)),  # deduplicate
            artifacts=artifacts,
            next_agents=next_agents,
        )