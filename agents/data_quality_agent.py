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

try:
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False


class DataQualityAgent(BaseAgent):
    name = "DataQualityAgent"

    # Thresholds
    MISSING_WARN  = 0.05   # 5 % missing → warning
    MISSING_HIGH  = 0.20   # 20 % missing → high severity
    DUP_WARN      = 0.01   # 1 % duplicates → warning

    def _run_drift_analysis(self, df: pd.DataFrame, reference_df) -> dict:
        """
        Compares df against reference_df using Evidently's DataDriftPreset.
        Returns drift_available=False (not an exception) if no reference
        dataset is supplied, Evidently isn't installed, or no columns are
        shared between the two frames — drift detection is an optional
        enhancement and must never break the rest of the agent.
        """
        if not _EVIDENTLY_AVAILABLE:
            return {"drift_available": False, "drift_error": "Evidently not installed."}
        if reference_df is None or reference_df.empty:
            return {"drift_available": False}

        shared_cols = [c for c in df.columns if c in reference_df.columns]
        if not shared_cols:
            return {
                "drift_available": False,
                "drift_error": "No shared columns between current and reference dataset.",
            }

        try:
            data_definition = DataDefinition()
            current_dataset = Dataset.from_pandas(df[shared_cols], data_definition=data_definition)
            reference_dataset = Dataset.from_pandas(reference_df[shared_cols], data_definition=data_definition)

            report = Report([DataDriftPreset()])
            result = report.run(reference_dataset, current_dataset)
            result_dict = result.dict()

            drifted_columns = []
            n_columns_checked = 0
            drift_share = None
            n_drifted_columns = None

            for metric in result_dict.get("metrics", []):
                metric_name = metric.get("metric_name", "")
                config = metric.get("config", {})
                value = metric.get("value")

                if metric_name.startswith("DriftedColumnsCount") and isinstance(value, dict):
                    n_drifted_columns = value.get("count")
                    drift_share = value.get("share")
                elif metric_name.startswith("ValueDrift"):
                    n_columns_checked += 1
                    col_name = config.get("column", "unknown")
                    threshold = config.get("threshold", 0.05)
                    if isinstance(value, (int, float)) and value < threshold:
                        drifted_columns.append(col_name)

            return {
                "drift_available": True,
                "drifted_columns": drifted_columns,
                "n_columns_checked": n_columns_checked or len(shared_cols),
                "n_drifted_columns": int(n_drifted_columns) if n_drifted_columns is not None else len(drifted_columns),
                "drift_share": round(drift_share, 3) if drift_share is not None else None,
            }
        except Exception as e:
            return {"drift_available": False, "drift_error": str(e)}

    def _run_trust_check(self, df: pd.DataFrame, reference_df) -> dict:
        """
        Runs an Evidently TestSuite (drift preset with include_tests=True)
        to produce a pass/fail trust score. Requires a reference dataset —
        unlike drift analysis, there is no meaningful "trust" check against
        a dataset compared to itself (it would always pass), so this
        honestly reports trust_available=False rather than faking a score
        when no reference exists.
        """
        if not _EVIDENTLY_AVAILABLE:
            return {
                "trust_available": False,
                "trust_tests_passed": None,
                "trust_tests_total": None,
                "trust_pass_rate": None,
                "trust_error": "Evidently not installed.",
            }
        if reference_df is None or reference_df.empty:
            return {
                "trust_available": False,
                "trust_tests_passed": None,
                "trust_tests_total": None,
                "trust_pass_rate": None,
            }

        shared_cols = [c for c in df.columns if c in reference_df.columns]
        if not shared_cols:
            return {
                "trust_available": False,
                "trust_tests_passed": None,
                "trust_tests_total": None,
                "trust_pass_rate": None,
                "trust_error": "No shared columns between current and reference dataset.",
            }

        try:
            data_definition = DataDefinition()
            current_dataset = Dataset.from_pandas(df[shared_cols], data_definition=data_definition)
            reference_dataset = Dataset.from_pandas(reference_df[shared_cols], data_definition=data_definition)

            report = Report([DataDriftPreset()], include_tests=True)
            result = report.run(reference_dataset, current_dataset)
            result_dict = result.dict()

            tests = result_dict.get("tests", [])
            passed = sum(1 for t in tests if t.get("status") == "SUCCESS")
            total = len(tests)

            return {
                "trust_available": True,
                "trust_tests_passed": passed,
                "trust_tests_total": total,
                "trust_pass_rate": round(passed / total, 3) if total else None,
            }
        except Exception as e:
            return {
                "trust_available": False,
                "trust_tests_passed": None,
                "trust_tests_total": None,
                "trust_pass_rate": None,
                "trust_error": str(e),
            }

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

        # ── 5. Drift detection & trust scoring (optional, needs reference) ─
        reference_df = (context or {}).get("drift_reference_df")
        drift_result = self._run_drift_analysis(df, reference_df)
        trust_result = self._run_trust_check(df, reference_df)

        artifacts["drift"] = drift_result
        artifacts["trust"] = trust_result

        if drift_result.get("drift_available"):
            drifted_cols = drift_result.get("drifted_columns", [])
            drift_share = drift_result.get("drift_share")
            if drifted_cols:
                sev = "high" if (drift_share or 0) >= 0.5 else "medium"
                findings.append({
                    "title": "Data drift detected vs. reference dataset",
                    "detail": (
                        f"{len(drifted_cols)} of {drift_result.get('n_columns_checked')} "
                        f"columns drifted ({', '.join(drifted_cols)})."
                    ),
                    "severity": sev,
                })
                recommendations.append(
                    "Investigate drifted columns — model performance may degrade "
                    "if drift is not addressed."
                )
        elif drift_result.get("drift_error"):
            # Don't surface "Evidently not installed" or "no reference dataset"
            # as a quality finding — those are setup states, not data issues.
            pass

        # ── 6. Quality score (simple heuristic) ───────────────────────────
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

        # Blend in Evidently trust pass rate when available, weighted at 30%
        # so it nudges rather than overrides the existing penalty-based score.
        if trust_result.get("trust_pass_rate") is not None:
            evidently_component = trust_result["trust_pass_rate"] * 100
            quality_score = round(quality_score * 0.7 + evidently_component * 0.3, 1)
            quality_score = max(0, min(100, quality_score))

        artifacts["quality_score"] = quality_score

        # ── 7. Status & summary ───────────────────────────────────────────
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
        if drift_result.get("drift_available"):
            n_drifted = len(drift_result.get("drifted_columns", []))
            summary += f" {n_drifted} column(s) show drift vs. reference dataset."

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