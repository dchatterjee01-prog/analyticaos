import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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
    </style>
    """, unsafe_allow_html=True)


# ── CHART COLOR PALETTE ──────────────────────────────────────────────────────
PALETTE = [
    PRIMARY_COLOR, ACCENT_COLOR, "#FFB347", "#64B5F6",
    "#CE93D8", "#FF6B6B", "#4ECDC4", "#FFD93D"
]


def _layout_theme(fig, title):
    fig.update_layout(
        title=title,
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(gridcolor="#2a2a45"),
        yaxis=dict(gridcolor="#2a2a45"),
        legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#2a2a45"),
        margin=dict(t=60, b=50),
        height=480,
    )
    return fig


# ── CHART BUILDERS ───────────────────────────────────────────────────────────
def _bar_chart(df, x, y, color):
    agg = df.groupby(x)[y].mean().reset_index() if color is None else None
    if color and color != "None":
        fig = px.bar(df, x=x, y=y, color=color,
                     color_discrete_sequence=PALETTE)
    else:
        fig = px.bar(df, x=x, y=y, color_discrete_sequence=[PRIMARY_COLOR])
    return _layout_theme(fig, f"Bar Chart — {y} by {x}")


def _line_chart(df, x, y, color):
    if color and color != "None":
        fig = px.line(df, x=x, y=y, color=color,
                       color_discrete_sequence=PALETTE)
    else:
        fig = px.line(df, x=x, y=y, color_discrete_sequence=[PRIMARY_COLOR])
    return _layout_theme(fig, f"Line Chart — {y} over {x}")


def _scatter_chart(df, x, y, color):
    if color and color != "None":
        fig = px.scatter(df, x=x, y=y, color=color,
                          color_discrete_sequence=PALETTE, opacity=0.7)
    else:
        fig = px.scatter(df, x=x, y=y, color_discrete_sequence=[PRIMARY_COLOR],
                          opacity=0.7)
    return _layout_theme(fig, f"Scatter Plot — {x} vs {y}")


def _pie_chart(df, names, values):
    fig = px.pie(df, names=names, values=values,
                 color_discrete_sequence=PALETTE, hole=0.35)
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title=f"Pie Chart — {values} by {names}",
        height=480,
    )
    return fig


def _histogram_chart(df, x, bins):
    fig = px.histogram(df, x=x, nbins=bins,
                        color_discrete_sequence=[PRIMARY_COLOR])
    return _layout_theme(fig, f"Histogram — {x}")


def _box_chart(df, x, y):
    if x and x != "None":
        fig = px.box(df, x=x, y=y, color_discrete_sequence=PALETTE)
    else:
        fig = px.box(df, y=y, color_discrete_sequence=[PRIMARY_COLOR])
    return _layout_theme(fig, f"Box Plot — {y}")


def _area_chart(df, x, y, color):
    if color and color != "None":
        fig = px.area(df, x=x, y=y, color=color,
                       color_discrete_sequence=PALETTE)
    else:
        fig = px.area(df, x=x, y=y, color_discrete_sequence=[PRIMARY_COLOR])
    return _layout_theme(fig, f"Area Chart — {y} over {x}")


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("📈 Visualization Engine")
    st.caption("Phase 4 — Interactive Chart Builder")

    df: pd.DataFrame = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    all_cols = df.columns.tolist()

    # ── Chart type selector ──
    st.markdown('<div class="section-header">1. Choose Chart Type</div>',
                unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Chart type:",
        options=[
            "Bar Chart", "Line Chart", "Scatter Plot",
            "Pie Chart", "Histogram", "Box Plot", "Area Chart"
        ],
        key="chart_type"
    )

    st.markdown('<div class="section-header">2. Configure Axes</div>',
                unsafe_allow_html=True)

    fig = None

    if chart_type == "Bar Chart":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis (category):", options=all_cols, key="bar_x")
        y = c2.selectbox("Y axis (value):", options=num_cols, key="bar_y")
        color = c3.selectbox("Color by (optional):",
                             options=["None"] + cat_cols, key="bar_color")
        if st.button("🎨 Generate Bar Chart", key="gen_bar"):
            fig = _bar_chart(df, x, y, color)

    elif chart_type == "Line Chart":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis:", options=all_cols, key="line_x")
        y = c2.selectbox("Y axis:", options=num_cols, key="line_y")
        color = c3.selectbox("Color by (optional):",
                             options=["None"] + cat_cols, key="line_color")
        if st.button("🎨 Generate Line Chart", key="gen_line"):
            fig = _line_chart(df, x, y, color)

    elif chart_type == "Scatter Plot":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis:", options=num_cols, key="scatter_x")
        y = c2.selectbox("Y axis:", options=num_cols,
                         index=min(1, len(num_cols)-1), key="scatter_y")
        color = c3.selectbox("Color by (optional):",
                             options=["None"] + cat_cols, key="scatter_color")
        if st.button("🎨 Generate Scatter Plot", key="gen_scatter"):
            fig = _scatter_chart(df, x, y, color)

    elif chart_type == "Pie Chart":
        c1, c2 = st.columns(2)
        names = c1.selectbox("Category column:", options=cat_cols or all_cols,
                             key="pie_names")
        values = c2.selectbox("Value column:", options=num_cols, key="pie_values")
        if st.button("🎨 Generate Pie Chart", key="gen_pie"):
            fig = _pie_chart(df, names, values)

    elif chart_type == "Histogram":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Column:", options=num_cols, key="hist_x")
        bins = c2.slider("Number of bins:", 5, 100, 30, key="hist_bins")
        if st.button("🎨 Generate Histogram", key="gen_hist"):
            fig = _histogram_chart(df, x, bins)

    elif chart_type == "Box Plot":
        c1, c2 = st.columns(2)
        y = c1.selectbox("Value column:", options=num_cols, key="box_y")
        x = c2.selectbox("Group by (optional):",
                         options=["None"] + cat_cols, key="box_x")
        if st.button("🎨 Generate Box Plot", key="gen_box"):
            fig = _box_chart(df, x, y)

    elif chart_type == "Area Chart":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis:", options=all_cols, key="area_x")
        y = c2.selectbox("Y axis:", options=num_cols, key="area_y")
        color = c3.selectbox("Color by (optional):",
                             options=["None"] + cat_cols, key="area_color")
        if st.button("🎨 Generate Area Chart", key="gen_area"):
            fig = _area_chart(df, x, y, color)

    # ── Display chart ──
    if fig is not None:
        st.markdown('<div class="section-header">3. Generated Chart</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, width='stretch')

        # ── Export options ──
        st.markdown('<div class="section-header">4. Export Chart</div>',
                    unsafe_allow_html=True)
        html_bytes = fig.to_html().encode("utf-8")
        st.download_button(
            label="⬇️ Download Chart (HTML — interactive)",
            data=html_bytes,
            file_name=f"{chart_type.replace(' ', '_').lower()}.html",
            mime="text/html",
            key="dl_chart_html"
        )
        # ════════════════════════════════════════════════════════════════
    # SECTION 2 — MULTI-CHART DASHBOARD BUILDER
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🖥️ Dashboard Builder</div>',
                unsafe_allow_html=True)

    st.info(
        "Build a multi-chart dashboard. Add charts one at a time — "
        "they will be arranged in a grid below."
    )

    # ── Initialize dashboard storage ──
    if "dashboard_charts" not in st.session_state:
        st.session_state["dashboard_charts"] = []

    # ── Add chart to dashboard ──
    st.markdown('<div class="section-header">Add Chart to Dashboard</div>',
                unsafe_allow_html=True)

    dash_chart_type = st.selectbox(
        "Chart type to add:",
        options=["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"],
        key="dash_chart_type"
    )

    dc1, dc2, dc3 = st.columns(3)

    if dash_chart_type == "Bar Chart":
        dx = dc1.selectbox("X axis:", options=all_cols, key="dash_bar_x")
        dy = dc2.selectbox("Y axis:", options=num_cols, key="dash_bar_y")
        if dc3.button("➕ Add to Dashboard", key="add_dash_bar"):
            new_fig = _bar_chart(df, dx, dy, "None")
            st.session_state["dashboard_charts"].append({
                "title": f"Bar — {dy} by {dx}",
                "fig": new_fig
            })
            st.success("✅ Chart added to dashboard.")

    elif dash_chart_type == "Line Chart":
        dx = dc1.selectbox("X axis:", options=all_cols, key="dash_line_x")
        dy = dc2.selectbox("Y axis:", options=num_cols, key="dash_line_y")
        if dc3.button("➕ Add to Dashboard", key="add_dash_line"):
            new_fig = _line_chart(df, dx, dy, "None")
            st.session_state["dashboard_charts"].append({
                "title": f"Line — {dy} over {dx}",
                "fig": new_fig
            })
            st.success("✅ Chart added to dashboard.")

    elif dash_chart_type == "Scatter Plot":
        dx = dc1.selectbox("X axis:", options=num_cols, key="dash_sc_x")
        dy = dc2.selectbox("Y axis:", options=num_cols,
                           index=min(1, len(num_cols)-1), key="dash_sc_y")
        if dc3.button("➕ Add to Dashboard", key="add_dash_sc"):
            new_fig = _scatter_chart(df, dx, dy, "None")
            st.session_state["dashboard_charts"].append({
                "title": f"Scatter — {dx} vs {dy}",
                "fig": new_fig
            })
            st.success("✅ Chart added to dashboard.")

    elif dash_chart_type == "Histogram":
        dx = dc1.selectbox("Column:", options=num_cols, key="dash_hist_x")
        dbins = dc2.slider("Bins:", 5, 100, 30, key="dash_hist_bins")
        if dc3.button("➕ Add to Dashboard", key="add_dash_hist"):
            new_fig = _histogram_chart(df, dx, dbins)
            st.session_state["dashboard_charts"].append({
                "title": f"Histogram — {dx}",
                "fig": new_fig
            })
            st.success("✅ Chart added to dashboard.")

    # ── Render dashboard grid ──
    charts = st.session_state["dashboard_charts"]

    if charts:
        st.markdown('<div class="section-header">📊 Your Dashboard</div>',
                    unsafe_allow_html=True)

        # 2-column grid
        for i in range(0, len(charts), 2):
            grid_cols = st.columns(2)
            for j, col in enumerate(grid_cols):
                idx = i + j
                if idx < len(charts):
                    with col:
                        st.plotly_chart(
                            charts[idx]["fig"],
                            width='stretch',
                            key=f"dash_chart_{idx}"
                        )
                        if st.button(
                            f"🗑️ Remove",
                            key=f"remove_dash_{idx}"
                        ):
                            st.session_state["dashboard_charts"].pop(idx)
                            st.rerun()

        # ── Clear all ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Entire Dashboard", key="clear_dashboard"):
            st.session_state["dashboard_charts"] = []
            st.rerun()

        # ── Export full dashboard as HTML ──
        st.markdown('<div class="section-header">Export Dashboard</div>',
                    unsafe_allow_html=True)

        dashboard_html = "<html><head><title>AnalyticaOS Dashboard</title></head><body style='background:#0F0F1A;'>"
        for c in charts:
            dashboard_html += f"<h2 style='color:#6C63FF;font-family:sans-serif;'>{c['title']}</h2>"
            dashboard_html += c["fig"].to_html(full_html=False, include_plotlyjs="cdn")
        dashboard_html += "</body></html>"

        st.download_button(
            label="⬇️ Download Full Dashboard (HTML)",
            data=dashboard_html.encode("utf-8"),
            file_name="analyticaos_dashboard.html",
            mime="text/html",
            key="dl_full_dashboard"
        )
        
    else:
        st.info("No charts added yet. Use the controls above to build your dashboard.")
        # ════════════════════════════════════════════════════════════════
    # SECTION 3 — ADVANCED CHART CUSTOMIZATION
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🎛️ Advanced Chart Customizer</div>',
                unsafe_allow_html=True)

    st.info(
        "Build a fully customized chart with trendlines, custom titles, "
        "color themes, and export-ready sizing."
    )

    # ── Step 1: Chart type + axes ──
    adv1, adv2, adv3 = st.columns(3)
    adv_chart_type = adv1.selectbox(
        "Chart type:",
        options=["Scatter Plot", "Line Chart", "Bar Chart"],
        key="adv_chart_type"
    )
    adv_x = adv2.selectbox("X axis:", options=all_cols, key="adv_x")
    adv_y = adv3.selectbox(
        "Y axis:", options=num_cols,
        index=min(1, len(num_cols)-1) if len(num_cols) > 1 else 0,
        key="adv_y"
    )

    # ── Step 2: Custom titles ──
    st.markdown('<div class="section-header">Custom Labels</div>',
                unsafe_allow_html=True)
    lab1, lab2, lab3 = st.columns(3)
    custom_title  = lab1.text_input("Chart title:", value=f"{adv_y} vs {adv_x}",
                                     key="adv_title")
    custom_xlabel = lab2.text_input("X axis label:", value=adv_x, key="adv_xlabel")
    custom_ylabel = lab3.text_input("Y axis label:", value=adv_y, key="adv_ylabel")

    # ── Step 3: Theme & size ──
    st.markdown('<div class="section-header">Theme & Size</div>',
                unsafe_allow_html=True)
    th1, th2, th3 = st.columns(3)
    color_theme = th1.selectbox(
        "Color theme:",
        options=["Indigo (Brand)", "Teal", "Sunset", "Ocean", "Mono"],
        key="adv_theme"
    )
    chart_height = th2.slider("Chart height (px):", 300, 800, 480, key="adv_height")
    show_trend   = th3.checkbox("Show trendline (numeric X/Y only)",
                                value=False, key="adv_trend")

    theme_colors = {
        "Indigo (Brand)": PRIMARY_COLOR,
        "Teal"           : ACCENT_COLOR,
        "Sunset"         : "#FF6B6B",
        "Ocean"          : "#64B5F6",
        "Mono"           : "#E8E8F0",
    }
    chosen_color = theme_colors[color_theme]

    # ── Step 4: Generate ──
    if st.button("🎨 Generate Custom Chart", key="gen_adv_chart"):
        adv_df = df.copy()

        if adv_chart_type == "Scatter Plot":
            fig_adv = go.Figure()
            fig_adv.add_trace(go.Scatter(
                x=adv_df[adv_x], y=adv_df[adv_y],
                mode="markers",
                marker=dict(color=chosen_color, size=6, opacity=0.7),
                name=custom_ylabel,
            ))

            if show_trend and pd.api.types.is_numeric_dtype(adv_df[adv_x]):
                z = adv_df[[adv_x, adv_y]].dropna()
                if len(z) > 1:
                    coeffs = pd.Series(
                        __import__("numpy").polyfit(z[adv_x], z[adv_y], 1)
                    )
                    trend_y = coeffs[0] * z[adv_x] + coeffs[1]
                    fig_adv.add_trace(go.Scatter(
                        x=z[adv_x], y=trend_y,
                        mode="lines",
                        line=dict(color=ACCENT_COLOR, width=2, dash="dash"),
                        name="Trendline",
                    ))

        elif adv_chart_type == "Line Chart":
            fig_adv = go.Figure()
            fig_adv.add_trace(go.Scatter(
                x=adv_df[adv_x], y=adv_df[adv_y],
                mode="lines",
                line=dict(color=chosen_color, width=2.5),
                name=custom_ylabel,
            ))

        else:  # Bar Chart
            fig_adv = go.Figure()
            fig_adv.add_trace(go.Bar(
                x=adv_df[adv_x], y=adv_df[adv_y],
                marker_color=chosen_color,
                opacity=0.85,
                name=custom_ylabel,
            ))

        fig_adv.update_layout(
            title=dict(text=custom_title, font=dict(size=20, color=PRIMARY_COLOR)),
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=SURFACE_COLOR,
            font_color=TEXT_COLOR,
            xaxis=dict(title=custom_xlabel, gridcolor="#2a2a45"),
            yaxis=dict(title=custom_ylabel, gridcolor="#2a2a45"),
            legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#2a2a45"),
            margin=dict(t=70, b=50),
            height=chart_height,
        )

        st.plotly_chart(fig_adv, width='stretch')

        # ── Export options ──
        st.markdown('<div class="section-header">Export Custom Chart</div>',
                    unsafe_allow_html=True)
        e1, e2 = st.columns(2)

        html_bytes_adv = fig_adv.to_html().encode("utf-8")
        e1.download_button(
            label="⬇️ Download HTML (interactive)",
            data=html_bytes_adv,
            file_name="custom_chart.html",
            mime="text/html",
            key="dl_adv_html"
        )

        try:
            png_bytes = fig_adv.to_image(format="png", width=1200, height=chart_height)
            e2.download_button(
                label="⬇️ Download PNG (static image)",
                data=png_bytes,
                file_name="custom_chart.png",
                mime="image/png",
                key="dl_adv_png"
            )
        except Exception:
            e2.info("PNG export requires `kaleido` package — install with: `pip install kaleido`")
            # ════════════════════════════════════════════════════════════════
    # SECTION 4 — CROSS-TAB HEATMAP & GEO VISUALIZATION
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🗺️ Cross-Tab Heatmap</div>',
                unsafe_allow_html=True)

    if len(cat_cols) < 1 or len(num_cols) < 1:
        st.info(
            "Need at least 1 categorical column and 1 numeric column "
            "for cross-tab heatmap."
        )
    else:
        st.markdown(
            "Visualize how a numeric value varies across two categorical "
            "dimensions — useful for finding hidden patterns."
        )

        ct1, ct2, ct3 = st.columns(3)

        cat_options = cat_cols if len(cat_cols) >= 2 else cat_cols + all_cols
        ct_row = ct1.selectbox("Row dimension:", options=cat_options,
                               key="ct_row")
        ct_col = ct2.selectbox(
            "Column dimension:",
            options=[c for c in cat_options if c != ct_row] or cat_options,
            key="ct_col"
        )
        ct_val = ct3.selectbox("Value (numeric):", options=num_cols,
                               key="ct_val")

        ct_agg = st.radio(
            "Aggregation:",
            options=["mean", "sum", "count", "median"],
            horizontal=True,
            key="ct_agg"
        )

        if st.button("🎨 Generate Cross-Tab Heatmap", key="gen_crosstab"):
            try:
                pivot = pd.pivot_table(
                    df, values=ct_val, index=ct_row, columns=ct_col,
                    aggfunc=ct_agg
                ).round(2)

                fig_ct = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.astype(str).tolist(),
                    y=pivot.index.astype(str).tolist(),
                    colorscale="Viridis",
                    text=pivot.values,
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    showscale=True,
                ))
                fig_ct.update_layout(
                    title=f"{ct_agg.capitalize()} of {ct_val} — {ct_row} × {ct_col}",
                    paper_bgcolor=BACKGROUND_COLOR,
                    plot_bgcolor=SURFACE_COLOR,
                    font_color=TEXT_COLOR,
                    height=max(400, len(pivot.index) * 40 + 100),
                    margin=dict(t=60, b=60, l=120),
                    xaxis=dict(tickangle=-35),
                )
                st.plotly_chart(fig_ct, width='stretch')

                with st.expander("📋 View Pivot Table Data"):
                    st.dataframe(pivot.astype(str), width='stretch')

            except Exception as e:
                st.error(f"Could not generate heatmap: {e}")

    # ════════════════════════════════════════════════════════════════
    # SECTION 5 — GEOGRAPHIC MAP (if lat/lon detected)
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🌍 Geographic Visualization</div>',
                unsafe_allow_html=True)

    # ── Auto-detect lat/lon columns ──
    lat_candidates = [c for c in all_cols if c.lower() in
                      ("lat", "latitude", "lat_", "y_coord")]
    lon_candidates = [c for c in all_cols if c.lower() in
                      ("lon", "lng", "longitude", "long", "x_coord")]

    if not lat_candidates or not lon_candidates:
        st.info(
            "⚪ No latitude/longitude columns detected. "
            "Geographic mapping requires columns named like "
            "'latitude' and 'longitude'."
        )
    else:
        geo1, geo2, geo3 = st.columns(3)
        lat_col = geo1.selectbox("Latitude column:", options=lat_candidates,
                                 key="geo_lat")
        lon_col = geo2.selectbox("Longitude column:", options=lon_candidates,
                                 key="geo_lon")
        size_col = geo3.selectbox(
            "Size by (optional):",
            options=["None"] + num_cols,
            key="geo_size"
        )

        if st.button("🗺️ Generate Map", key="gen_geo_map"):
            geo_df = df.dropna(subset=[lat_col, lon_col])

            fig_geo = px.scatter_mapbox(
                geo_df,
                lat=lat_col,
                lon=lon_col,
                size=size_col if size_col != "None" else None,
                color_discrete_sequence=[ACCENT_COLOR],
                zoom=3,
                height=550,
            )
            fig_geo.update_layout(
                mapbox_style="carto-darkmatter",
                paper_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_geo, width='stretch')