"""
Stage K — SQL Analytics Agent.
Full-database scan: row counts, null rates, distinct counts,
numeric summaries, top-N frequency tables.
Follows the existing BaseAgent / AgentResult contract.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from agents.base_agent import BaseAgent, AgentResult
from connections.db_engine import (
    get_engine,
    get_conn_meta,
    list_tables,
    describe_table,
    table_row_count,
    run_query,
)

_TOP_N       = 5     # top-N values per categorical column
_MAX_TABLES  = 50    # cap full-DB scan to avoid timeout on huge DBs
_NULL_WARN   = 0.20  # null rate threshold for medium severity
_NULL_HIGH   = 0.50  # null rate threshold for high severity
_CARD_WARN   = 0.95  # cardinality ratio threshold (nearly unique → likely key)


# ── Per-column result ─────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name:          str
    dtype:         str
    null_count:    int
    null_rate:     float
    distinct_count: int
    cardinality:   float        # distinct / total rows
    top_values:    list[dict]   # [{"value": x, "count": n}, …]
    min_val:       str  = ""
    max_val:       str  = ""
    mean_val:      str  = ""


# ── Per-table result ──────────────────────────────────────────────────────────

@dataclass
class TableProfile:
    name:        str
    row_count:   int
    col_count:   int
    columns:     list[ColumnProfile] = field(default_factory=list)
    warnings:    list[str]           = field(default_factory=list)


# ── Agent ─────────────────────────────────────────────────────────────────────

class SQLAnalyticsAgent(BaseAgent):
    """
    Full-database scan agent.
    Profiles every table (up to _MAX_TABLES) in the connected database,
    generates findings and recommendations, returns an AgentResult.
    """

    def run(self, **kwargs) -> AgentResult:
        engine = get_engine()
        if engine is None:
            return AgentResult(
                agent_name="SQLAnalyticsAgent",
                status="error",
                summary="No active database connection.",
                findings=[],
                recommendations=["Connect to a database via SQL Connect first."],
                artifacts={},
                next_agents=[],
            )

        meta   = get_conn_meta()
        tables = list_tables()

        if not tables:
            return AgentResult(
                agent_name="SQLAnalyticsAgent",
                status="error",
                summary="No tables found in the connected database.",
                findings=[],
                recommendations=["Verify the database user has SELECT on at least one table."],
                artifacts={},
                next_agents=[],
            )

        scanned      = tables[:_MAX_TABLES]
        skipped      = len(tables) - len(scanned)
        profiles:    list[TableProfile] = []
        findings     = []
        recommendations = []

        for tbl in scanned:
            profile = self._profile_table(tbl)
            profiles.append(profile)

            # ── Table-level findings ──────────────────────────────────────────
            if profile.row_count == 0:
                findings.append({
                    "severity": "medium",
                    "table": tbl,
                    "message": f"Table `{tbl}` is empty (0 rows).",
                })

            for col in profile.columns:
                # High null rate
                if col.null_rate >= _NULL_HIGH:
                    findings.append({
                        "severity": "high",
                        "table": tbl,
                        "column": col.name,
                        "message": (
                            f"`{tbl}`.`{col.name}`: "
                            f"{col.null_rate:.0%} null — consider dropping or imputing."
                        ),
                    })
                elif col.null_rate >= _NULL_WARN:
                    findings.append({
                        "severity": "medium",
                        "table": tbl,
                        "column": col.name,
                        "message": (
                            f"`{tbl}`.`{col.name}`: "
                            f"{col.null_rate:.0%} null — review before analysis."
                        ),
                    })

                # Near-unique column (likely a key / ID)
                if (col.cardinality >= _CARD_WARN
                        and profile.row_count > 10
                        and col.distinct_count > 1):
                    findings.append({
                        "severity": "low",
                        "table": tbl,
                        "column": col.name,
                        "message": (
                            f"`{tbl}`.`{col.name}`: "
                            f"cardinality {col.cardinality:.0%} — likely a key column."
                        ),
                    })

        # ── Cross-table recommendations ───────────────────────────────────────
        empty_tables = [p.name for p in profiles if p.row_count == 0]
        if empty_tables:
            recommendations.append(
                f"Empty tables detected: {', '.join(empty_tables)}. "
                "Exclude from analysis or investigate data pipeline."
            )

        high_null_cols = [
            f"`{f['table']}`.`{f['column']}`"
            for f in findings
            if f["severity"] == "high" and "column" in f
        ]
        if high_null_cols:
            recommendations.append(
                f"{len(high_null_cols)} column(s) have ≥50% nulls: "
                f"{', '.join(high_null_cols[:5])}."
                + (" (and more…)" if len(high_null_cols) > 5 else "")
            )

        if skipped > 0:
            recommendations.append(
                f"{skipped} table(s) skipped — database exceeds "
                f"{_MAX_TABLES}-table scan limit. "
                "Run the agent on a subset schema for full coverage."
            )

        # ── Summary ───────────────────────────────────────────────────────────
        total_rows = sum(p.row_count for p in profiles)
        summary = (
            f"Scanned {len(scanned)} table(s) in **{meta.get('database','')}** "
            f"({meta.get('dialect_label','DB')}). "
            f"Total rows across all tables: {total_rows:,}. "
            f"{len(findings)} finding(s) detected."
        )

        # ── Artifacts — DataFrames for UI rendering ───────────────────────────
        artifacts = {
            "profiles":      profiles,          # list[TableProfile]
            "table_summary": self._summary_df(profiles),  # pd.DataFrame
        }

        return AgentResult(
            agent_name="SQLAnalyticsAgent",
            status="success",
            summary=summary,
            findings=findings,
            recommendations=recommendations,
            artifacts=artifacts,
            next_agents=["EDAAgent", "ReportAgent"],
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _profile_table(self, table_name: str) -> TableProfile:
        try:
            desc_df   = describe_table(table_name)
            row_count = table_row_count(table_name)
            col_names = desc_df["name"].tolist()
            col_types = dict(zip(desc_df["name"], desc_df["type"]))
        except Exception:
            return TableProfile(
                name=table_name, row_count=0, col_count=0,
                warnings=[f"Could not introspect `{table_name}`."]
            )

        col_profiles = []
        for col in col_names:
            cp = self._profile_column(table_name, col,
                                      col_types.get(col, ""), row_count)
            col_profiles.append(cp)

        return TableProfile(
            name=table_name,
            row_count=row_count,
            col_count=len(col_names),
            columns=col_profiles,
        )

    def _profile_column(
        self, table: str, col: str, dtype: str, row_count: int
    ) -> ColumnProfile:
        safe_col   = f"`{col}`"
        safe_table = f"`{table}`"

        # Null count + distinct count in one query
        try:
            df = run_query(
                f"SELECT "
                f"  COUNT(*) - COUNT({safe_col}) AS null_count, "
                f"  COUNT(DISTINCT {safe_col}) AS distinct_count "
                f"FROM {safe_table}",
                limit=1,
            )
            null_count    = int(df.iloc[0]["null_count"])
            distinct_count = int(df.iloc[0]["distinct_count"])
        except Exception:
            null_count    = 0
            distinct_count = 0

        null_rate    = null_count / row_count if row_count > 0 else 0.0
        cardinality  = distinct_count / row_count if row_count > 0 else 0.0

        # Top-N values
        top_values = []
        try:
            df_top = run_query(
                f"SELECT {safe_col} AS value, COUNT(*) AS cnt "
                f"FROM {safe_table} "
                f"WHERE {safe_col} IS NOT NULL "
                f"GROUP BY {safe_col} "
                f"ORDER BY cnt DESC "
                f"LIMIT {_TOP_N}",
                limit=_TOP_N,
            )
            top_values = df_top.to_dict(orient="records")
        except Exception:
            pass

        # Numeric min/max/mean
        min_val = max_val = mean_val = ""
        dtype_upper = dtype.upper()
        is_numeric  = any(t in dtype_upper for t in [
            "INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL", "BIGINT"
        ])
        if is_numeric:
            try:
                df_stats = run_query(
                    f"SELECT "
                    f"  MIN({safe_col}) AS min_val, "
                    f"  MAX({safe_col}) AS max_val, "
                    f"  AVG({safe_col}) AS mean_val "
                    f"FROM {safe_table}",
                    limit=1,
                )
                min_val  = str(df_stats.iloc[0]["min_val"])
                max_val  = str(df_stats.iloc[0]["max_val"])
                mean_val = f"{float(df_stats.iloc[0]['mean_val']):.4f}"
            except Exception:
                pass

        return ColumnProfile(
            name=col,
            dtype=dtype,
            null_count=null_count,
            null_rate=null_rate,
            distinct_count=distinct_count,
            cardinality=cardinality,
            top_values=top_values,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
        )

    def _summary_df(self, profiles: list[TableProfile]) -> pd.DataFrame:
        """Build a cross-table summary DataFrame for the UI."""
        rows = []
        for p in profiles:
            high = sum(
                1 for c in p.columns
                if c.null_rate >= _NULL_HIGH
            )
            med = sum(
                1 for c in p.columns
                if _NULL_WARN <= c.null_rate < _NULL_HIGH
            )
            rows.append({
                "table":        p.name,
                "rows":         p.row_count,
                "columns":      p.col_count,
                "high_null_cols": high,
                "med_null_cols":  med,
            })
        return pd.DataFrame(rows)