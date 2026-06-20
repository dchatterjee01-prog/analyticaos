"""
anomaly_agent.py
Multivariate anomaly detection using sklearn's IsolationForest.
Lets the user (or Orchestrator) flag rows that look statistically
unusual across a set of numeric columns, rather than the per-column
IQR/Z-Score logic already in pages/eda.py (Phase 3).

Follows the same BaseAgent/AgentResult pattern as Phase 9's agents
so the Orchestrator can call it interchangeably.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from agents.base_agent import BaseAgent, AgentResult


class AnomalyAgent(BaseAgent):
    name = "AnomalyAgent"

    MIN_ROWS         = 10     # below this, detection is unreliable
    DEFAULT_CONTAM   = 0.05   # assume ~5% of rows are anomalous by default
    HIGH_ANOMALY_PCT = 0.15   # if >15% flagged, something's likely wrong with contamination/columns

    def run(self, df: pd.DataFrame, context: dict | None = None) -> AgentResult:
        context = context or {}
        findings        = []
        recommendations = []
        artifacts        = {}

        # ── 1. Resolve which columns to use ───────────────────────────────
        cols = context.get("anomaly_cols")
        if cols is None:
            cols = df.select_dtypes(include="number").columns.tolist()
            if not cols:
                return AgentResult(
                    agent_name=self.name,
                    status="error",
                    summary="No numeric columns available for anomaly detection.",
                )
        else:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                return AgentResult(
                    agent_name=self.name,
                    status="error",
                    summary=f"Requested columns not found: {', '.join(missing)}",
                )

        contamination = context.get("contamination", self.DEFAULT_CONTAM)
        contamination = max(0.001, min(0.5, contamination))  # sklearn valid range guard

        # ── 2. Prepare data ────────────────────────────────────────────────
        work_df = df[cols].copy()
        work_df = work_df.apply(pd.to_numeric, errors="coerce")

        valid_mask = work_df.notna().all(axis=1)
        clean_df = work_df[valid_mask]

        if len(clean_df) < self.MIN_ROWS:
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary=(
                    f"Only {len(clean_df)} complete rows across selected columns — "
                    f"need at least {self.MIN_ROWS} for reliable detection."
                ),
            )

        dropped_rows = len(df) - len(clean_df)
        if dropped_rows > 0:
            findings.append({
                "title":    "Rows excluded due to missing values",
                "detail":   f"{dropped_rows} rows had nulls in selected columns and were excluded.",
                "severity": "low",
            })

        # ── 3. Scale + fit IsolationForest ─────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean_df)

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        predictions = model.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
        scores = model.decision_function(X_scaled)  # higher = more normal

        is_anomaly = predictions == -1
        n_anomalies = int(is_anomaly.sum())
        anomaly_pct = n_anomalies / len(clean_df)

        # ── 4. Build result frame (aligned back to original df index) ─────
        result_df = clean_df.copy()
        result_df["anomaly_score"] = -scores  # flip sign: higher = more anomalous
        result_df["is_anomaly"] = is_anomaly
        artifacts["anomaly_results"] = result_df
        artifacts["flagged_indices"] = clean_df.index[is_anomaly].tolist()
        artifacts["columns_used"] = cols
        artifacts["contamination_used"] = contamination
        artifacts["dropped_rows"] = dropped_rows

        # ── 5. PCA projection for visualization (2D) ────────────────────────
        if len(cols) >= 2:
            n_components = min(2, X_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(
                coords,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=clean_df.index,
            )
            pca_df["is_anomaly"] = is_anomaly
            artifacts["pca_projection"] = pca_df
            artifacts["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()

        # ── 6. Top contributing columns per anomaly (simple z-score based) ─
        if n_anomalies > 0:
            z_scores = np.abs((clean_df - clean_df.mean()) / clean_df.std(ddof=0).replace(0, np.nan))
            anomaly_z = z_scores[is_anomaly]
            top_contributors = anomaly_z.mean().sort_values(ascending=False).head(3)
            artifacts["top_contributing_columns"] = top_contributors.round(3).to_dict()

            findings.append({
                "title":    "Most anomaly-driving columns",
                "detail":   f"Highest average deviation: {', '.join(top_contributors.index.tolist())}",
                "severity": "low",
            })

        # ── 7. Findings on volume of anomalies ──────────────────────────────
        if anomaly_pct >= self.HIGH_ANOMALY_PCT:
            findings.append({
                "title":    "Unusually high anomaly rate",
                "detail":   f"{n_anomalies} rows ({anomaly_pct*100:.1f}%) flagged — "
                            f"consider lowering contamination or reviewing column choice.",
                "severity": "medium",
            })
            recommendations.append(
                "Review selected columns or reduce the contamination parameter — "
                "current anomaly rate is higher than typically expected."
            )
        else:
            findings.append({
                "title":    "Anomalies detected",
                "detail":   f"{n_anomalies} rows ({anomaly_pct*100:.1f}%) flagged as anomalous "
                            f"across {len(cols)} column(s).",
                "severity": "low" if anomaly_pct < 0.10 else "medium",
            })

        if n_anomalies > 0:
            recommendations.append(
                f"Review the {n_anomalies} flagged rows — export via the anomaly results table."
            )

        # ── 8. Summary & status ─────────────────────────────────────────────
        status = "warning" if anomaly_pct >= self.HIGH_ANOMALY_PCT else "ok"
        summary = (
            f"Detected {n_anomalies} anomalies ({anomaly_pct*100:.1f}%) across "
            f"{len(cols)} numeric column(s) using IsolationForest "
            f"(contamination={contamination})."
        )

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary,
            findings=findings,
            recommendations=list(dict.fromkeys(recommendations)),
            artifacts=artifacts,
            next_agents=[],
        )