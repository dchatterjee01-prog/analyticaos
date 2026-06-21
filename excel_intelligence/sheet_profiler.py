"""
sheet_profiler.py
Reads every sheet in a multi-sheet Excel workbook, profiles each one
(row count, column dtypes, cardinality, candidate-key columns), and
classifies each sheet as a likely FACT or DIMENSION table using simple,
explainable heuristics — no ML, just row-count + cardinality signals.

Uses polars internally for fast per-column cardinality/null scans on
potentially large sheets, but returns pandas DataFrames for the rest of
the app (which is pandas-only throughout).
"""
from dataclasses import dataclass, field
import pandas as pd
import polars as pl


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    n_nulls: int
    null_pct: float
    is_likely_key: bool          # high-cardinality, low-null, often named *_id


@dataclass
class SheetProfile:
    sheet_name: str
    df: pd.DataFrame
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile] = field(default_factory=list)
    candidate_key_cols: list[str] = field(default_factory=list)
    role: str = "unknown"        # "fact" | "dimension" | "unknown"
    role_reason: str = ""


# ── Heuristic thresholds ───────────────────────────────────────────────────
KEY_UNIQUENESS_THRESHOLD = 0.95   # column is a candidate key if >=95% unique
KEY_NULL_THRESHOLD        = 0.02  # candidate keys should have near-zero nulls
DIMENSION_MAX_ROWS_RATIO  = 0.3   # dimension tables are usually much smaller
KEY_NAME_HINTS = ("id", "code", "key", "no", "number")


def _profile_column(pl_df: pl.DataFrame, col: str, n_rows: int) -> ColumnProfile:
    series = pl_df[col]
    n_unique = series.n_unique()
    n_nulls = series.null_count()
    null_pct = (n_nulls / n_rows) if n_rows else 0.0
    uniqueness = (n_unique / n_rows) if n_rows else 0.0
    dtype_str = str(series.dtype)
    dtype_lower = dtype_str.lower()

    name_lower = col.lower()
    has_key_name_hint = any(hint in name_lower for hint in KEY_NAME_HINTS)

    # Continuous float columns (revenue, price, measurements, etc.) can be
    # accidentally near-100%-unique just from random/measured variation —
    # that does NOT make them identifier columns. Real keys are integers,
    # strings, or explicitly hinted by name (e.g. 'order_id', 'sku_code').
    is_float_dtype = dtype_lower.startswith("float")
    is_key_eligible_dtype = not is_float_dtype

    is_likely_key = (
        is_key_eligible_dtype
        and uniqueness >= KEY_UNIQUENESS_THRESHOLD
        and null_pct <= KEY_NULL_THRESHOLD
        and n_rows > 1
        and (has_key_name_hint or "int" in dtype_lower)
    )

    return ColumnProfile(
        name=col,
        dtype=dtype_str,
        n_unique=n_unique,
        n_nulls=n_nulls,
        null_pct=round(null_pct * 100, 2),
        is_likely_key=is_likely_key,
    )


def _classify_role(profile: SheetProfile, all_row_counts: list[int]) -> tuple[str, str]:
    """
    Classifies a sheet as fact or dimension using simple relative heuristics:
      - FACT tables tend to be the largest sheet(s) and have multiple
        candidate-key-like columns pointing to OTHER tables (foreign keys),
        but usually lack a single dominant unique identifier of their own.
      - DIMENSION tables tend to be smaller, have exactly one strong
        candidate primary key, and often have descriptive text columns.
    """
    if not all_row_counts:
        return "unknown", "No comparison data available."

    max_rows = max(all_row_counts)
    relative_size = profile.n_rows / max_rows if max_rows else 0

    n_key_like_cols = sum(1 for c in profile.columns if c.is_likely_key)
    has_single_strong_key = n_key_like_cols == 1
    has_multiple_key_like = n_key_like_cols >= 2

    text_cols = sum(
        1 for c in profile.columns
        if c.dtype in ("String", "Utf8") and not c.is_likely_key
    )

    if relative_size >= 0.8 and len(all_row_counts) > 1:
        return "fact", (
            f"Largest sheet by row count ({profile.n_rows:,} rows, "
            f"{relative_size*100:.0f}% of the biggest sheet's size) — "
            f"typical of a transactional fact table."
        )

    if has_single_strong_key and relative_size <= DIMENSION_MAX_ROWS_RATIO and text_cols >= 1:
        key_col = next(c.name for c in profile.columns if c.is_likely_key)
        return "dimension", (
            f"Small relative size ({relative_size*100:.0f}% of largest sheet) "
            f"with one clear primary key ('{key_col}') and descriptive "
            f"text columns — typical of a lookup/dimension table."
        )

    if has_multiple_key_like and relative_size >= 0.4:
        return "fact", (
            f"Contains {n_key_like_cols} key-like columns "
            f"(likely foreign keys referencing other sheets) with no single "
            f"dominant identifier — typical of a fact table."
        )

    if has_single_strong_key:
        key_col = next(c.name for c in profile.columns if c.is_likely_key)
        return "dimension", (
            f"Has one clear primary key ('{key_col}') — "
            f"treated as a dimension table by default."
        )

    return "unknown", (
        "No strong fact or dimension signal found — "
        "manual review recommended."
    )


def profile_sheet(sheet_name: str, df: pd.DataFrame) -> SheetProfile:
    """Profiles a single sheet. Returns a SheetProfile with role left as
    'unknown' — role classification happens after all sheets are profiled,
    since it depends on relative comparison across sheets."""
    n_rows, n_cols = df.shape

    # Sanitize before handing to polars — mixed-type object columns
    # (a known pandas/PyArrow gotcha already documented in this project)
    # cause polars ingestion errors otherwise.
    safe_df = df.copy()
    for col in safe_df.columns:
        if safe_df[col].dtype == object:
            safe_df[col] = safe_df[col].astype(str)

    try:
        pl_df = pl.from_pandas(safe_df)
    except Exception:
        # Fallback: if polars still chokes on something unusual, profile
        # using pandas-native calls instead of failing the whole sheet.
        pl_df = None

    columns = []
    candidate_keys = []

    for col in df.columns:
        if pl_df is not None:
            cp = _profile_column(pl_df, col, n_rows)
        else:
            series = df[col]
            n_unique = series.nunique()
            n_nulls = series.isnull().sum()
            null_pct = (n_nulls / n_rows * 100) if n_rows else 0.0
            uniqueness = (n_unique / n_rows) if n_rows else 0.0
            name_lower = col.lower()
            has_hint = any(h in name_lower for h in KEY_NAME_HINTS)
            is_float_dtype = pd.api.types.is_float_dtype(series)
            is_int_dtype = pd.api.types.is_integer_dtype(series)
            cp = ColumnProfile(
                name=col, dtype=str(series.dtype),
                n_unique=n_unique, n_nulls=n_nulls, null_pct=round(null_pct, 2),
                is_likely_key=(
                    not is_float_dtype
                    and uniqueness >= KEY_UNIQUENESS_THRESHOLD
                    and null_pct <= KEY_NULL_THRESHOLD * 100
                    and (has_hint or is_int_dtype)
                ),
            )
        columns.append(cp)
        if cp.is_likely_key:
            candidate_keys.append(col)

    return SheetProfile(
        sheet_name=sheet_name,
        df=df,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        candidate_key_cols=candidate_keys,
    )


def profile_workbook(sheets: dict[str, pd.DataFrame]) -> list[SheetProfile]:
    """
    Profiles every sheet in a workbook (dict of sheet_name -> DataFrame,
    as returned by pd.read_excel(..., sheet_name=None)), then classifies
    each sheet's role relative to the others.
    """
    profiles = [profile_sheet(name, df) for name, df in sheets.items()]
    all_row_counts = [p.n_rows for p in profiles]

    for profile in profiles:
        role, reason = _classify_role(profile, all_row_counts)
        profile.role = role
        profile.role_reason = reason

    return profiles
