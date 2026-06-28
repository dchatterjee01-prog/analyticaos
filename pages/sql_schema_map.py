"""
Stage O — Schema Relationship Mapper page.
Detects FK relationships and renders a static SVG ER diagram.
Requires an active DB connection from pages/sql_connect.py.
"""

import streamlit as st
from connections.db_engine import get_engine, get_conn_meta
from connections.schema_mapper import (
    extract_schema_map,
    generate_svg,
    schema_stats,
    SchemaMap,
)

# ── Page config ───────────────────────────────────────────────────────────────


def show():
    st.set_page_config(
        page_title="Schema Map", page_icon="🗺️", layout="wide"
    )
    st.title("🗺️ Schema Relationship Mapper")
    st.caption(
        "Detects foreign key relationships and renders an ER-style schema diagram."
    )

    # ── Guard: must be connected ──────────────────────────────────────────────────
    if not get_engine():
        st.warning(
            "⚠️ No active database connection. "
            "Please connect first via **SQL Connect**."
        )
        st.stop()

    meta = get_conn_meta()
    st.info(
        f"Connected to **{meta['database']}** "
        f"({meta.get('dialect_label', 'DB')}) — read-only"
    )

    # ── Cache key ─────────────────────────────────────────────────────────────────
    _MAP_KEY = "schema_map_result"
    _SVG_KEY = "schema_map_svg"

    # ── Controls ──────────────────────────────────────────────────────────────────
    col_scan, col_clear, _ = st.columns([1, 1, 4])
    scan_clicked  = col_scan.button("🔍 Scan Schema")
    clear_clicked = col_clear.button("🗑 Clear")

    if clear_clicked:
        st.session_state.pop(_MAP_KEY, None)
        st.session_state.pop(_SVG_KEY, None)
        st.rerun()

    if scan_clicked:
        with st.spinner("Inspecting schema and detecting relationships…"):
            try:
                schema_map = extract_schema_map()
                svg        = generate_svg(schema_map)
                st.session_state[_MAP_KEY] = schema_map
                st.session_state[_SVG_KEY] = svg
            except Exception as e:
                st.error(f"Schema scan failed: {e}")

    # ── Render ────────────────────────────────────────────────────────────────────
    if _MAP_KEY not in st.session_state:
        st.info(
            "Click **Scan Schema** to detect tables, columns, "
            "primary keys, and foreign key relationships."
        )
        st.stop()

    schema_map: SchemaMap = st.session_state[_MAP_KEY]
    svg:        str       = st.session_state[_SVG_KEY]

    # Warnings (e.g. tables that could not be inspected)
    if schema_map.warnings:
        with st.expander("⚠️ Warnings", expanded=False):
            for w in schema_map.warnings:
                st.warning(w)

    # ── Metrics row ───────────────────────────────────────────────────────────────
    st.divider()
    stats = schema_stats(schema_map)

    col_t, col_c, col_r, col_pk, col_fk = st.columns(5)
    col_t.metric("Tables",        stats["tables"])
    col_c.metric("Columns",       stats["columns"])
    col_r.metric("Relationships", stats["relationships"])
    col_pk.metric("PK Columns",   stats["pk_columns"])
    col_fk.metric("FK Columns",   stats["fk_columns"])

    # ── Legend ────────────────────────────────────────────────────────────────────
    st.caption(
        "🔑 Primary Key &nbsp;|&nbsp; "
        "🔗 Foreign Key &nbsp;|&nbsp; "
        "─ ─ ─ FK relationship arrow"
    )

    # ── SVG diagram ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📐 ER Diagram")

    if not schema_map.tables:
        st.info("No tables found — nothing to diagram.")
    else:
        # Render SVG inline via st.html (Streamlit 1.31+)
        # Wrap in a scrollable div for large schemas
        scrollable_svg = (
            "<div style='"
            "overflow:auto; "
            "border:1px solid #2A5C8C; "
            "border-radius:8px; "
            "padding:12px; "
            "background:#F7F9FC;"
            "'>"
            + svg
            + "</div>"
        )
        st.html(scrollable_svg)

        # SVG download
        st.download_button(
            label="⬇️ Download SVG",
            data=svg.encode("utf-8"),
            file_name=f"{meta['database']}_schema.svg",
            mime="image/svg+xml",
            key="schema_svg_download",
        )

    # ── Relationship table ────────────────────────────────────────────────────────
    if schema_map.relationships:
        st.divider()
        st.subheader("🔗 Foreign Key Relationships")
        import pandas as pd
        rel_rows = [
            {
                "From Table":  r.from_table,
                "From Column": r.from_col,
                "To Table":    r.to_table,
                "To Column":   r.to_col,
            }
            for r in schema_map.relationships
        ]
        rel_df = pd.DataFrame(rel_rows)
        st.dataframe(rel_df.astype(str), use_container_width=True)
        st.caption(f"{len(schema_map.relationships)} FK relationship(s) detected.")
    else:
        st.divider()
        st.info(
            "No foreign key relationships detected. "
            "This may be expected if the database uses application-level FK enforcement "
            "rather than database-level constraints."
        )

    # ── Per-table detail ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Per-Table Column Details")

    for tbl in schema_map.tables:
        pk_count = sum(1 for c in tbl.columns if c.is_pk)
        fk_count = sum(1 for c in tbl.columns if c.is_fk)

        with st.expander(
            f"**{tbl.name}** — "
            f"{len(tbl.columns)} columns · "
            f"{pk_count} PK · {fk_count} FK",
            expanded=False,
        ):
            import pandas as pd
            col_rows = [
                {
                    "Column":      c.name,
                    "Type":        c.dtype,
                    "Primary Key": "🔑" if c.is_pk else "",
                    "Foreign Key": "🔗" if c.is_fk else "",
                    "References":  c.fk_ref if c.fk_ref else "",
                }
                for c in tbl.columns
            ]
            col_df = pd.DataFrame(col_rows)
            st.dataframe(col_df.astype(str), use_container_width=True)