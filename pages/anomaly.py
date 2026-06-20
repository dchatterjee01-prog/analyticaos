import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from agents import AnomalyAgent
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR,
    TEXT_COLOR, BACKGROUND_COLOR
)

# ── CSS ──────────────────────────────────────────────────────────────────────
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
    .metric-card {{
        background: {SURFACE_COLOR};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #2a2a45;
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {ACCENT_COLOR};
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: {TEXT_COLOR};
        opacity: 0.75;
        margin-top: 0.2rem;
    }}
    .clean-badge {{
        background: {ACCENT_COLOR}22;
        border: 1px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .warn-badge {{
        background: #FF6B6B22;
        border: 1px solid #FF6B6B;
        color: #FF6B6B;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)


def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    """Converts '#RRGGBB' to 'rgba(r,g,b,a)' — Plotly does not accept
    CSS-style '#RRGGBBAA' 8-digit hex (see Phase 11.5 Step Group A)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── PCA SCATTER PLOT ──────────────────────────────────────────────────────────
def _pca_scatter(pca_df: pd.DataFrame, explained_var: list):
    normal = pca_df[~pca_df["is_anomaly"]]
    anomalies = pca_df[pca_df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["PC1"], y=normal["PC2"],
        mode="markers", name="Normal",
        marker=dict(color=_hex_to_rgba(PRIMARY_COLOR, 0.55), size=7),
    ))
    fig.add_trace(go.Scatter(
        x=anomalies["PC1"], y=anomalies["PC2"],
        mode="markers", name="Anomaly",
        marker=dict(
            color="#FF6B6B", size=11, symbol="x",
            line=dict(width=1.5, color="#FF6B6B"),
        ),
    ))

    var_pc1 = explained_var[0] * 100 if len(explained_var) > 0 else 0
    var_pc2 = explained_var[1] * 100 if len(explained_var) > 1 else 0

    fig.update_layout(
        title="Anomalies — PCA Projection (2D)",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(title=f"PC1 ({var_pc1:.1f}% variance)", gridcolor="#2a2a45"),
        yaxis=dict(title=f"PC2 ({var_pc2:.1f}% variance)", gridcolor="#2a2a45"),
        legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#2a2a45"),
        margin=dict(t=50, b=50),
        height=460,
    )
    return fig


# ── TOP CONTRIBUTORS BAR CHART ────────────────────────────────────────────────
def _contributors_bar(contributors: dict):
    cols = list(contributors.keys())
    vals = list(contributors.values())

    fig = go.Figure(go.Bar(
        x=vals, y=cols, orientation="h",
        marker_color=ACCENT_COLOR, opacity=0.85,
    ))
    fig.update_layout(
        title="Top Anomaly-Driving Columns (avg. z-score among anomalies)",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(title="Average |z-score|", gridcolor="#2a2a45"),
        yaxis=dict(gridcolor="#2a2a45"),
        margin=dict(t=50, b=40, l=120),
        height=300,
    )
    return fig


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("🚨 Anomaly Detection")
    st.caption("Phase 11.5 — Multivariate Anomaly Detection (IsolationForest)")

    df: pd.DataFrame = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns available for anomaly detection.")
        return

    # ── Setup ──
    st.markdown('<div class="section-header">Detection Setup</div>',
                unsafe_allow_html=True)

    selected_cols = st.multiselect(
        "Columns to include in anomaly detection:",
        options=num_cols,
        default=num_cols[:min(5, len(num_cols))],
        key="anomaly_cols_select",
    )

    contamination = st.slider(
        "Expected anomaly rate (contamination):",
        min_value=0.01, max_value=0.30, value=0.05, step=0.01,
        key="anomaly_contamination",
        help="Roughly what fraction of rows you expect to be anomalous. "
             "Higher values flag more rows as anomalies."
    )

    run_clicked = st.button("▶️ Run Anomaly Detection", type="primary", width="stretch")

    if run_clicked:
        if len(selected_cols) < 1:
            st.warning("⚠️ Select at least one column.")
            return

        with st.spinner("Running IsolationForest..."):
            agent = AnomalyAgent()
            result = agent.run(df, context={
                "anomaly_cols": selected_cols,
                "contamination": contamination,
            })

        st.session_state["anomaly_result"] = result

    if "anomaly_result" not in st.session_state:
        st.info("Click **Run Anomaly Detection** to start.")
        return

    result = st.session_state["anomaly_result"]

    if result.status == "error":
        st.error(f"❌ {result.summary}")
        return

    # ── Metrics row ──
    st.markdown('<div class="section-header">Detection Summary</div>',
                unsafe_allow_html=True)

    artifacts = result.artifacts
    n_anomalies = len(artifacts.get("flagged_indices", []))
    total_rows = len(artifacts.get("anomaly_results", pd.DataFrame()))
    pct = (n_anomalies / total_rows * 100) if total_rows else 0

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, f"{total_rows:,}", "Rows Analyzed")
    _metric_card(m2, f"{n_anomalies:,}", "Anomalies Flagged")
    _metric_card(m3, f"{pct:.1f}%", "Anomaly Rate")
    _metric_card(m4, f"{artifacts.get('contamination_used', contamination):.2f}", "Contamination Used")

    if result.status == "warning":
        st.markdown(
            '<span class="warn-badge">⚠️ Anomaly rate higher than typically expected — '
            'review column choice or contamination setting</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="clean-badge">✅ Detection completed normally</span>',
            unsafe_allow_html=True
        )

    # ── Findings ──
    if result.findings:
        st.markdown('<div class="section-header">Findings</div>',
                    unsafe_allow_html=True)
        for f in result.findings:
            st.markdown(f"- **{f['title']}** — {f['detail']}")

    # ── PCA visualization ──
    if "pca_projection" in artifacts:
        st.markdown('<div class="section-header">Visualization</div>',
                    unsafe_allow_html=True)
        fig = _pca_scatter(artifacts["pca_projection"], artifacts.get("pca_explained_variance", []))
        st.plotly_chart(fig, width="stretch")
    elif len(selected_cols) < 2:
        st.info("ℹ️ Select 2 or more columns to see the PCA scatter visualization.")

    # ── Top contributors ──
    if artifacts.get("top_contributing_columns"):
        st.plotly_chart(
            _contributors_bar(artifacts["top_contributing_columns"]),
            width="stretch"
        )

    # ── Results table + download ──
    st.markdown('<div class="section-header">Flagged Rows</div>',
                unsafe_allow_html=True)

    results_df = artifacts.get("anomaly_results")
    if results_df is not None:
        flagged_only = st.checkbox("Show only flagged anomalies", value=True, key="anomaly_filter")
        display_df = results_df[results_df["is_anomaly"]] if flagged_only else results_df
        display_df = display_df.sort_values("anomaly_score", ascending=False)

        st.dataframe(display_df.astype(str), width="stretch")

        csv = display_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="⬇️ Download Flagged Rows (CSV)",
            data=csv,
            file_name="anomaly_flagged_rows.csv",
            mime="text/csv",
            key="dl_anomaly",
        )