"""
relationship_finder.py
Given a list of SheetProfile objects (from sheet_profiler.py), infers
likely join relationships between sheets and builds a directed graph
(networkx) representing fact -> dimension foreign-key relationships.

Matching strategy (explainable, no ML):
  1. Name similarity — a column in a fact-like sheet matches a candidate
     key column in a dimension-like sheet if their names are equal, or
     one is a clear variant of the other (e.g. 'customer_id' in Orders
     matching 'customer_id' or 'id' in Customers).
  2. Dtype compatibility — both columns must be comparable types
     (both integer-like, or both string-like).
  3. Value overlap — the fact-side column's values must substantially
     overlap with the dimension-side key's actual values (not just
     name/dtype matching, which can produce false positives).
"""
from dataclasses import dataclass, field
import re
import pandas as pd
import networkx as nx

from excel_intelligence.sheet_profiler import SheetProfile


@dataclass
class InferredRelationship:
    from_sheet: str        # typically the fact/child sheet
    from_column: str
    to_sheet: str           # typically the dimension/parent sheet
    to_column: str
    match_basis: str        # "exact_name" | "normalized_name" | "value_overlap"
    overlap_pct: float      # % of from_column's non-null values found in to_column
    confidence: str         # "high" | "medium" | "low"


MIN_OVERLAP_FOR_RELATIONSHIP = 0.70   # at least 70% of FK values must exist in PK
HIGH_CONFIDENCE_OVERLAP      = 0.95


def _normalize_col_name(name: str) -> str:
    """Strips common suffixes/prefixes so 'customer_id' and 'id' (in a
    sheet called 'Customers') can be recognized as the same concept."""
    n = name.lower().strip()
    n = re.sub(r"[_\-\s]+", "_", n)
    n = re.sub(r"^(id_|key_)", "", n)
    n = re.sub(r"(_id|_key|_no|_code)$", "", n)
    return n


def _dtype_compatible(dtype_a: str, dtype_b: str) -> bool:
    # Case-insensitive on purpose: depending on whether a sheet went
    # through the polars path or the pandas fallback path (see
    # sheet_profiler.py — polars requires pyarrow for some column
    # combinations and silently falls back otherwise), dtype strings can
    # come back as either 'Int64' (polars) or 'int64' (pandas) for the
    # same underlying integer data. A case-sensitive substring check
    # caused a real bug here: real customer_id<->customer_id matches
    # were silently dropped because "Int" not in "int64".
    dtype_a_lower = dtype_a.lower()
    dtype_b_lower = dtype_b.lower()

    int_like = ("int",)
    str_like = ("string", "utf8", "object", "str")

    a_is_int = any(t in dtype_a_lower for t in int_like)
    b_is_int = any(t in dtype_b_lower for t in int_like)
    a_is_str = any(t in dtype_a_lower for t in str_like)
    b_is_str = any(t in dtype_b_lower for t in str_like)

    return (a_is_int and b_is_int) or (a_is_str and b_is_str)


def _compute_overlap(fact_series: pd.Series, dim_series: pd.Series) -> float:
    """Returns the fraction of fact_series's non-null values that exist
    in dim_series's value set."""
    fact_vals = fact_series.dropna()
    if len(fact_vals) == 0:
        return 0.0
    dim_set = set(dim_series.dropna().tolist())
    if not dim_set:
        return 0.0
    matches = fact_vals.isin(dim_set).sum()
    return matches / len(fact_vals)


def _sheet_name_hint_matches(col_name: str, sheet_name: str) -> bool:
    """Checks if a column name plausibly refers to a given sheet, e.g.
    'customer_id' referring to a sheet named 'Customers'."""
    normalized_col = _normalize_col_name(col_name)
    sheet_singular = sheet_name.lower().rstrip("s")
    return normalized_col in sheet_singular or sheet_singular in normalized_col


def find_relationships(profiles: list[SheetProfile]) -> list[InferredRelationship]:
    """
    Searches for plausible foreign-key relationships between every pair
    of sheets. Only considers columns from the 'fact'-or-'unknown'-role
    sheet matching against candidate KEY columns of 'dimension'-role
    sheets, since that's the conceptually correct direction (fact
    references dimension, not the other way around).
    """
    relationships: list[InferredRelationship] = []

    dimension_profiles = [p for p in profiles if p.role == "dimension"]
    other_profiles = [p for p in profiles if p.role != "dimension"]

    if not dimension_profiles:
        return relationships

    for fact_profile in other_profiles:
        for fact_col in fact_profile.columns:
            normalized_fact_col = _normalize_col_name(fact_col.name)

            for dim_profile in dimension_profiles:
                if dim_profile.sheet_name == fact_profile.sheet_name:
                    continue

                for dim_key_col_name in dim_profile.candidate_key_cols:
                    dim_col_profile = next(
                        (c for c in dim_profile.columns if c.name == dim_key_col_name),
                        None
                    )
                    if dim_col_profile is None:
                        continue

                    normalized_dim_col = _normalize_col_name(dim_key_col_name)

                    name_matches = (
                        fact_col.name == dim_key_col_name
                        or normalized_fact_col == normalized_dim_col
                        or _sheet_name_hint_matches(fact_col.name, dim_profile.sheet_name)
                    )
                    if not name_matches:
                        continue

                    if not _dtype_compatible(fact_col.dtype, dim_col_profile.dtype):
                        continue

                    overlap = _compute_overlap(
                        fact_profile.df[fact_col.name],
                        dim_profile.df[dim_key_col_name],
                    )

                    if overlap < MIN_OVERLAP_FOR_RELATIONSHIP:
                        continue

                    match_basis = (
                        "exact_name" if fact_col.name == dim_key_col_name
                        else "normalized_name"
                    )
                    confidence = "high" if overlap >= HIGH_CONFIDENCE_OVERLAP else "medium"

                    relationships.append(InferredRelationship(
                        from_sheet=fact_profile.sheet_name,
                        from_column=fact_col.name,
                        to_sheet=dim_profile.sheet_name,
                        to_column=dim_key_col_name,
                        match_basis=match_basis,
                        overlap_pct=round(overlap * 100, 1),
                        confidence=confidence,
                    ))

    return relationships


def build_relationship_graph(
    profiles: list[SheetProfile],
    relationships: list[InferredRelationship],
) -> nx.DiGraph:
    """Builds a directed graph: edge from fact-sheet -> dimension-sheet
    for every inferred relationship. Node attributes carry role + row
    count for visualization purposes (Step 3)."""
    G = nx.DiGraph()

    for profile in profiles:
        G.add_node(
            profile.sheet_name,
            role=profile.role,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
        )

    for rel in relationships:
        G.add_edge(
            rel.from_sheet,
            rel.to_sheet,
            from_column=rel.from_column,
            to_column=rel.to_column,
            confidence=rel.confidence,
            overlap_pct=rel.overlap_pct,
        )

    return G
