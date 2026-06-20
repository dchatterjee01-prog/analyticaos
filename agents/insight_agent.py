"""
insight_agent.py
Wraps Phase 3 (EDA) + Phase 6 (Stats) logic into a callable agent.
Produces plain-language insight bullets from:
  - descriptive stats (numeric cols)
  - top correlations
  - skewness flags
  - high-cardinality categoricals
  - normality quick-check (Shapiro on small samples, skew/kurt heuristic on large)
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from agents.base_agent import BaseAgent, AgentResult


class InsightAgent(BaseAgent):
    name = "InsightAgent"

    CORR_STRONG   = 0.70
    CORR_MODERATE = 0.40
    SKEW_THRESH   = 1.0
    HIGH_CARD     = 50     # unique values threshold for "high cardinality"
    SHAPIRO_MAX_N = 5000   # use Shapiro-Wilk only for n ≤ this

    def run(self, df: pd.DataFrame, context: dict | None = None) -> AgentResult:
        findings        = []
        recommendations = []
        artifacts       = {}

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if not num_cols and not cat_cols:
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary="No usable columns found for insight generation.",
            )

        # ── 1. Descriptive stats ──────────────────────────────────────────
        if num_cols:
            desc = df[num_cols].describe().T
            artifacts["descriptive_stats"] = desc

        # ── 2. Skewness ───────────────────────────────────────────────────
        skewed_cols = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            skew = series.skew()
            if abs(skew) >= self.SKEW_THRESH:
                direction = "right" if skew > 0 else "left"
                skewed_cols.append(col)
                sev = "medium" if abs(skew) >= 2 else "low"
                findings.append({
                    "title":    f"Skewed distribution: '{col}'",
                    "detail":   f"Skewness = {skew:.2f} ({direction}-skewed). Mean ≠ Median.",
                    "severity": sev,
                })
                if abs(skew) >= 2:
                    recommendations.append(
                        f"Consider log or Box-Cox transform on '{col}' before modelling."
                    )

        artifacts["skewed_cols"] = skewed_cols

        # ── 3. Correlations ───────────────────────────────────────────────
        strong_pairs  = []
        moderate_pairs = []
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            artifacts["correlation_matrix"] = corr_matrix
            seen = set()
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i+1:]:
                    key = tuple(sorted([c1, c2]))
                    if key in seen:
                        continue
                    seen.add(key)
                    val = corr_matrix.loc[c1, c2]
                    if pd.isna(val):
                        continue
                    abs_val = abs(val)
                    direction = "positively" if val > 0 else "negatively"
                    if abs_val >= self.CORR_STRONG:
                        strong_pairs.append((c1, c2, round(val, 3)))
                        findings.append({
                            "title":    f"Strong correlation: '{c1}' ↔ '{c2}'",
                            "detail":   f"r = {val:.3f} ({direction} correlated).",
                            "severity": "low",   # not a problem, just an insight
                        })
                    elif abs_val >= self.CORR_MODERATE:
                        moderate_pairs.append((c1, c2, round(val, 3)))

            artifacts["strong_correlations"]   = strong_pairs
            artifacts["moderate_correlations"] = moderate_pairs

            if strong_pairs:
                recommendations.append(
                    "Review strongly correlated feature pairs before ML to avoid multicollinearity."
                )

        # ── 4. Normality quick-check ──────────────────────────────────────
        non_normal = []
        for col in num_cols:
            series = df[col].dropna()
            n = len(series)
            if n < 4:
                continue
            if n <= self.SHAPIRO_MAX_N:
                _, p = sp_stats.shapiro(series.sample(min(n, 5000), random_state=42))
                is_normal = p >= 0.05
                method = "Shapiro-Wilk"
            else:
                # heuristic: |skew| < 0.5 and |kurt| < 1
                skew  = series.skew()
                kurt  = series.kurt()
                is_normal = abs(skew) < 0.5 and abs(kurt) < 1
                method = "skew/kurtosis heuristic"

            if not is_normal:
                non_normal.append(col)
                findings.append({
                    "title":    f"Non-normal distribution: '{col}'",
                    "detail":   f"Normality rejected by {method}.",
                    "severity": "low",
                })

        artifacts["non_normal_cols"] = non_normal
        if non_normal:
            recommendations.append(
                "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis) for non-normal columns."
            )

        # ── 5. High-cardinality categoricals ─────────────────────────────
        high_card = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique >= self.HIGH_CARD:
                high_card.append(col)
                findings.append({
                    "title":    f"High-cardinality column: '{col}'",
                    "detail":   f"{n_unique} unique values — likely an ID or free-text field.",
                    "severity": "medium",
                })
                recommendations.append(
                    f"Exclude '{col}' from grouping/encoding; use as identifier only."
                )

        artifacts["high_cardinality_cols"] = high_card

        # ── 6. Summary ────────────────────────────────────────────────────
        status = "warning" if any(f["severity"] in ("medium","high") for f in findings) else "ok"
        summary = (
            f"Analysed {len(num_cols)} numeric + {len(cat_cols)} categorical columns. "
            f"{len(skewed_cols)} skewed, {len(strong_pairs)} strong correlation pair(s), "
            f"{len(non_normal)} non-normal distribution(s) found."
        )

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary,
            findings=findings,
            recommendations=list(dict.fromkeys(recommendations)),
            artifacts=artifacts,
            next_agents=["ModelingAgent"],
        )