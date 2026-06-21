import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from excel_intelligence import profile_workbook, find_relationships, build_relationship_graph
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR, TEXT_COLOR,
    BACKGROUND_COLOR, BORDER_COLOR, MUTED_COLOR,
    SUCCESS_COLOR, SUCCESS_BG, WARNING_COLOR, WARNING_BG,
)

ROLE_COLORS = {
    "fact": PRIMARY_COLOR,
    "dimension": SUCCESS_COLOR,
    "unknown": MUTED_COLOR,
}


def _inject_css():
    st.markdown(f"""
    <style>
    .section-header {{
        color: {PRIMARY_COLOR};
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 0.6rem;
    }}
    .sheet-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }}
    .sheet-name {{
        font-weight: 700;
        font-size: 1rem;
        color: {TEXT_COLOR};
    }}
    .role-badge-fact {{
        background: {PRIMARY_COLOR}1A;
        border: 1px solid {PRIMARY_COLOR};
        color: {PRIMARY_COLOR};
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }}
    .role-badge-dimension {{
        background: {SUCCESS_BG};
        border: 1px solid {SUCCESS_COLOR};
        color: {SUCCESS_COLOR};
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }}
    .role-badge-unknown {{
        background: {WARNING_BG};
        border: 1px solid {WARNING_COLOR};
        color: {WARNING_COLOR};
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }}
    .sheet-detail {{
        font-size: 0.82rem;
        color: {MUTED_COLOR};
        margin-top: 0.3rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def _role_badge_html(role: str) -> str:
    label = role.upper()
    cls = f"role-badge-{role}" if role in ROLE_COLORS else "role-badge-unknown"
    return f'<span class="{cls}">{label}</span>'


def _render_sheet_cards(profiles):
    for p in profiles:
        keys_str = ", ".join(p.candidate_key_cols) if p.candidate_key_cols else "none detected"
        st.markdown(f"""
        <div class="sheet-card">
            <span class="sheet-name">{p.sheet_name}</span> &nbsp; {_role_badge_html(p.role)}
            <div class="sheet-detail">
                {p.n_rows:,} rows × {p.n_cols} columns · Candidate keys: {keys_str}
            </div>
            <div class="sheet-detail">{p.role_reason}</div>
        </div>
        """, unsafe_allow_html=True)


def _build_network_figure(G: nx.DiGraph) -> go.Figure:
    if len(G.nodes) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=42, k=1.2)

    edge_x, edge_y, edge_labels_x, edge_labels_y, edge_labels_text = [], [], [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_labels_x.append((x0 + x1) / 2)
        edge_labels_y.append((y0 + y1) / 2)
        edge_labels_text.append(
            f"{data.get('from_column','')} → {data.get('to_column','')}\n"
            f"({data.get('confidence','')}, {data.get('overlap_pct',0)}%)"
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5, color=BORDER_COLOR),
        hoverinfo="none", showlegend=False,
    )

    edge_label_trace = go.Scatter(
        x=edge_labels_x, y=edge_labels_y, mode="text",
        text=["🔗" for _ in edge_labels_text],
        hovertext=edge_labels_text, hoverinfo="text",
        textfont=dict(size=14), showlegend=False,
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        role = data.get("role", "unknown")
        node_text.append(
            f"{node}<br>{data.get('n_rows',0):,} rows · {data.get('n_cols',0)} cols<br>role: {role}"
        )
        node_color.append(ROLE_COLORS.get(role, MUTED_COLOR))
        node_size.append(max(20, min(55, 20 + data.get("n_rows", 0) ** 0.3)))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="bottom center",
        hovertext=node_text, hoverinfo="text",
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color="white")),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, edge_label_trace, node_trace])
    fig.update_layout(
        title="Inferred Sheet Relationships",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=50, b=20, l=20, r=20),
        height=420,
    )
    return fig


def _build_merge_preview(profiles, selected_rels, anchor_sheet: str) -> tuple:
    """Merges the anchor (fact) sheet with each selected dimension sheet
    via the chosen relationships. Returns (merged_df, error_message)."""
    profile_by_name = {p.sheet_name: p for p in profiles}
    if anchor_sheet not in profile_by_name:
        return None, f"Anchor sheet '{anchor_sheet}' not found."

    merged = profile_by_name[anchor_sheet].df.copy()

    for rel in selected_rels:
        if rel.from_sheet != anchor_sheet:
            continue
        dim_df = profile_by_name[rel.to_sheet].df
        suffix = f"_{rel.to_sheet.lower()}"
        try:
            merged = merged.merge(
                dim_df,
                left_on=rel.from_column,
                right_on=rel.to_column,
                how="left",
                suffixes=("", suffix),
            )
        except Exception as e:
            return None, f"Merge failed for {rel.from_sheet}->{rel.to_sheet}: {e}"

    return merged, None


def show():
    _inject_css()
    st.title("🗂️ Multi-Sheet Excel Intelligence")
    st.caption("Stage B — Auto-detect fact/dimension sheets, infer relationships, build a merged dataset")

    uploaded = st.file_uploader(
        "Upload a multi-sheet Excel workbook (.xlsx)",
        type=["xlsx"],
        key="excel_intel_uploader",
    )

    if uploaded is None:
        st.info("Upload a workbook with 2 or more sheets to begin.")
        return

    with st.spinner("Reading and profiling all sheets..."):
        try:
            sheets = pd.read_excel(uploaded, sheet_name=None, engine="openpyxl")
        except Exception as e:
            st.error(f"❌ Could not read workbook: {e}")
            return

        if len(sheets) < 2:
            st.warning(
                "⚠️ This workbook has only 1 sheet — use the standard "
                "**📁 Upload Data** page instead for single-sheet files."
            )
            return

        for name in sheets:
            for col in sheets[name].columns:
                if sheets[name][col].dtype == object:
                    sheets[name][col] = sheets[name][col].astype(str)

        profiles = profile_workbook(sheets)
        relationships = find_relationships(profiles)
        graph = build_relationship_graph(profiles, relationships)

    st.session_state["excel_intel_profiles"] = profiles
    st.session_state["excel_intel_relationships"] = relationships

    # ── Sheet cards ──
    st.markdown('<div class="section-header">Detected Sheets</div>', unsafe_allow_html=True)
    _render_sheet_cards(profiles)

    # ── Relationship graph ──
    st.markdown('<div class="section-header">Relationship Graph</div>', unsafe_allow_html=True)
    if relationships:
        st.plotly_chart(_build_network_figure(graph), width="stretch")
        st.caption("Hover over the 🔗 markers on edges to see join column + confidence details.")
    else:
        st.info(
            "ℹ️ No relationships were confidently inferred. "
            "This can happen if sheets don't share recognizable key columns, "
            "or if value overlap is below the 70% threshold."
        )

    # ── Merge builder ──
    if relationships:
        st.markdown('<div class="section-header">Build Merged Dataset</div>', unsafe_allow_html=True)

        fact_sheets = [p.sheet_name for p in profiles if p.role in ("fact", "unknown")]
        if not fact_sheets:
            st.info("No fact-like sheet detected to use as a merge anchor.")
            return

        anchor = st.selectbox("Anchor sheet (the base table to merge others into):", fact_sheets)

        available_rels = [r for r in relationships if r.from_sheet == anchor]
        if not available_rels:
            st.info(f"No inferred relationships originate from '{anchor}'.")
            return

        rel_labels = [
            f"{r.from_sheet}.{r.from_column} → {r.to_sheet}.{r.to_column} "
            f"({r.confidence}, {r.overlap_pct}%)"
            for r in available_rels
        ]
        selected_idx = st.multiselect(
            "Select which relationships to merge in:",
            options=list(range(len(available_rels))),
            default=list(range(len(available_rels))),
            format_func=lambda i: rel_labels[i],
        )
        selected_rels = [available_rels[i] for i in selected_idx]

        if st.button("🔀 Preview Merge", type="primary", width="stretch"):
            merged_df, error = _build_merge_preview(profiles, selected_rels, anchor)
            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state["excel_intel_merge_preview"] = merged_df
                st.success(f"✅ Merge preview ready — {merged_df.shape[0]:,} rows × {merged_df.shape[1]} columns")

        if "excel_intel_merge_preview" in st.session_state:
            preview_df = st.session_state["excel_intel_merge_preview"]
            st.dataframe(preview_df.head(20).astype(str), width="stretch")

            if st.button("✅ Use This Merged Dataset", width="stretch"):
                st.session_state["df"] = preview_df
                st.session_state["df_original"] = preview_df.copy()
                st.session_state["filename"] = f"{uploaded.name} (merged: {anchor} + {len(selected_rels)} sheet(s))"
                st.success(
                    "✅ Merged dataset is now active across the app. "
                    "Navigate to any other page to continue."
                )