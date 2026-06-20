# pages/pivot.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR,
    TEXT_COLOR, ACCENT_COLOR
)


def _inject_css():
    st.markdown(f"""
    <style>
      .pivot-header {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.2rem;
      }}
      .pivot-sub {{
        font-size: 0.82rem;
        color: {TEXT_COLOR}88;
        margin-bottom: 1.2rem;
      }}
      .metric-card {{
        background: {SURFACE_COLOR};
        border: 1px solid {PRIMARY_COLOR}33;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
      }}
      .metric-val {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {ACCENT_COLOR};
      }}
      .metric-lbl {{
        font-size: 0.72rem;
        color: {TEXT_COLOR}88;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
      }}
    </style>
    """, unsafe_allow_html=True)


def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val">{value}</div>
      <div class="metric-lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def show():
    _inject_css()

    st.markdown('<div class="pivot-header">📊 Pivot Table Engine</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="pivot-sub">Summarize and rank your data like Excel Pivot Tables — instantly.</div>',
                unsafe_allow_html=True)

    df = st.session_state["df"].copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    all_cols     = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(
                       include=["object", "category"]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found. Use Data Cleaning → "
                 "Data Type Fixer first.")
        return

    # ── Render Tabs ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Pivot Builder", "🏆 Top N Rankings",
        "📈 Time Intelligence", "🥧 Contribution & Pareto"
    ])

    with tab1:
        _pivot_builder(df, numeric_cols, cat_cols, all_cols)

    with tab2:
        _top_n_rankings(df, numeric_cols, all_cols)

    with tab3:
        _time_intelligence(df, numeric_cols)

    with tab4:
        _contribution_pareto(df, numeric_cols, all_cols)


# ── Tab 1: Pivot Builder ─────────────────────────────────────────────────────

def _pivot_builder(df, numeric_cols, cat_cols, all_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Pivot Table")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        row_field = st.selectbox("📋 Rows", options=all_cols, index=0,
                                 key="pb_rows")
    with c2:
        col_options = ["(None)"] + all_cols
        col_field = st.selectbox("📰 Columns", options=col_options, index=0,
                                 key="pb_cols")
    with c3:
        value_field = st.selectbox("🔢 Values", options=numeric_cols,
                                   index=0, key="pb_vals")
    with c4:
        agg_func = st.selectbox(
            "∑ Aggregation",
            options=["sum", "mean", "count", "median", "min", "max"],
            index=0, key="pb_agg"
        )

    try:
        pivot_col = None if col_field == "(None)" else col_field
        pivot_df = pd.pivot_table(
            df, values=value_field, index=row_field,
            columns=pivot_col, aggfunc=agg_func, fill_value=0
        )
        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [" | ".join(str(c) for c in col)
                                 for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
    except Exception as e:
        st.error(f"Could not build pivot table: {e}")
        return

    st.divider()

    numeric_pivot = pivot_df.select_dtypes(include="number")
    total_val  = numeric_pivot.values.sum()
    max_val    = numeric_pivot.values.max()
    min_val    = numeric_pivot.values.min()
    row_count  = len(pivot_df)

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, f"{row_count:,}",       "Pivot Rows")
    _metric_card(m2, f"{total_val:,.1f}",    f"Total {agg_func}({value_field})")
    _metric_card(m3, f"{max_val:,.1f}",      "Max Value")
    _metric_card(m4, f"{min_val:,.1f}",      "Min Value")

    st.markdown("")

    st.markdown("### 📋 Pivot Table")
    display_df = pivot_df.copy()
    for col in display_df.select_dtypes(include="object").columns:
        display_df[col] = display_df[col].astype(str)
    st.dataframe(display_df, width="stretch", height=380)

    st.divider()

    if pivot_col is not None:
        st.markdown("### 🌡️ Pivot Heatmap")
        heatmap_df  = pivot_df.set_index(row_field)
        heatmap_num = heatmap_df.select_dtypes(include="number")
        if not heatmap_num.empty:
            fig = px.imshow(
                heatmap_num, color_continuous_scale="Viridis",
                aspect="auto",
                title=f"{agg_func}({value_field}) by {row_field} × {pivot_col}",
                text_auto=".1f"
            )
            fig.update_layout(
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR,
                title_font_color=PRIMARY_COLOR,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.markdown("### 📊 Bar Chart")
        num_cols_pivot = [c for c in pivot_df.columns if c != row_field]
        if num_cols_pivot:
            fig = px.bar(
                pivot_df, x=row_field, y=num_cols_pivot[0],
                color_discrete_sequence=[PRIMARY_COLOR],
                title=f"{agg_func}({value_field}) by {row_field}"
            )
            fig.update_layout(
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR,
                title_font_color=PRIMARY_COLOR,
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
                margin=dict(t=60, b=40, l=40, r=40)
            )
            st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### 💾 Export")
    dc1, dc2 = st.columns(2)

    csv_bytes = pivot_df.to_csv(index=False).encode("utf-8")
    dc1.download_button("⬇️ Download CSV", data=csv_bytes,
                        file_name="pivot_table.csv", mime="text/csv",
                        width="stretch")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        pivot_df.to_excel(writer, index=False, sheet_name="Pivot")
    dc2.download_button(
        "⬇️ Download Excel", data=buffer.getvalue(),
        file_name="pivot_table.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch"
    )


# ── Tab 2: Top N Rankings ────────────────────────────────────────────────────

def _top_n_rankings(df, numeric_cols, all_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Rankings")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        group_col = st.selectbox("📋 Group By", options=all_cols,
                                 index=0, key="tn_group")
    with r2:
        value_col = st.selectbox("🔢 Rank By", options=numeric_cols,
                                 index=0, key="tn_val")
    with r3:
        agg_fn = st.selectbox("∑ Aggregation",
                              options=["sum", "mean", "count",
                                       "median", "max", "min"],
                              index=0, key="tn_agg")
    with r4:
        top_n = st.slider("🏆 Top N", min_value=3,
                          max_value=50, value=10, step=1)

    order = st.radio("Order", options=["Top (Highest First)",
                                       "Bottom (Lowest First)"],
                     horizontal=True, key="tn_order")
    ascending = order.startswith("Bottom")

    try:
        ranked = (
            df.groupby(group_col)[value_col]
            .agg(agg_fn)
            .reset_index()
            .rename(columns={value_col: f"{agg_fn}({value_col})"})
            .sort_values(f"{agg_fn}({value_col})", ascending=ascending)
            .head(top_n)
            .reset_index(drop=True)
        )
        ranked.index += 1
        ranked.index.name = "Rank"
    except Exception as e:
        st.error(f"Could not compute rankings: {e}")
        return

    st.divider()

    val_col_name = f"{agg_fn}({value_col})"
    total = ranked[val_col_name].sum()
    top_1 = ranked[val_col_name].iloc[0]
    avg   = ranked[val_col_name].mean()

    m1, m2, m3 = st.columns(3)
    _metric_card(m1, f"{top_n}",          "Entries Shown")
    _metric_card(m2, f"{top_1:,.2f}",     "Top Entry Value")
    _metric_card(m3, f"{avg:,.2f}",       f"Avg {agg_fn}({value_col})")

    st.markdown("")
    st.markdown("### 🏆 Rankings Table")

    display = ranked.copy()
    for col in display.select_dtypes(include="object").columns:
        display[col] = display[col].astype(str)
    st.dataframe(display, width="stretch", height=380)

    st.divider()
    st.markdown("### 📊 Horizontal Bar Chart")

    fig = px.bar(
        ranked.reset_index(),
        x=val_col_name,
        y=group_col,
        orientation="h",
        color=val_col_name,
        color_continuous_scale="Viridis",
        title=f"Top {top_n}: {agg_fn}({value_col}) by {group_col}",
        text=val_col_name
    )
    fig.update_traces(texttemplate="%{text:,.1f}", textposition="outside")
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title_font_color=PRIMARY_COLOR,
        yaxis=dict(autorange="reversed", gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        coloraxis_showscale=False,
        margin=dict(t=60, b=40, l=40, r=80),
        height=max(350, top_n * 38)
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### 💾 Export Rankings")

    csv_bytes = ranked.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Rankings CSV", data=csv_bytes,
                       file_name="top_n_rankings.csv", mime="text/csv",
                       width="stretch")


# ── Tab 3: Time Intelligence ─────────────────────────────────────────────────

def _time_intelligence(df, numeric_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Time Intelligence")

    date_cols = []
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            date_cols.append(col)
        elif df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], format="mixed",
                                           errors="coerce")
                if converted.notna().sum() / len(df) > 0.7:
                    date_cols.append(col)
            except Exception:
                pass

    if not date_cols:
        st.warning("⚠️ No date columns detected. "
                   "Use **Data Cleaning → Data Type Fixer** to convert "
                   "a column to datetime first.")
        return

    if not numeric_cols:
        st.error("No numeric columns found.")
        return

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        date_col = st.selectbox("📅 Date Column", options=date_cols,
                                key="ti_date")
    with t2:
        value_col = st.selectbox("🔢 Value Column", options=numeric_cols,
                                 key="ti_val")
    with t3:
        agg_fn = st.selectbox("∑ Aggregation",
                              options=["sum", "mean", "count",
                                       "median", "max", "min"],
                              key="ti_agg")
    with t4:
        period = st.selectbox("🗓️ Time Period",
                              options=["Year", "Quarter",
                                       "Month", "Week"],
                              index=2, key="ti_period")

    show_growth = st.checkbox("Show Period-over-Period Growth Rate (%)",
                              value=True, key="ti_growth")

    try:
        work = df[[date_col, value_col]].copy()
        work[date_col] = pd.to_datetime(work[date_col],
                                        format="mixed", errors="coerce")
        work = work.dropna(subset=[date_col])

        period_map = {
            "Year":    "Y",
            "Quarter": "QE",
            "Month":   "ME",
            "Week":    "W"
        }
        freq = period_map[period]

        work = work.set_index(date_col)
        agg_df = work.resample(freq)[value_col].agg(agg_fn).reset_index()
        agg_df.columns = ["Period", "Value"]
        agg_df = agg_df.dropna(subset=["Value"])

        if period == "Year":
            agg_df["Label"] = agg_df["Period"].dt.strftime("%Y")
        elif period == "Quarter":
            agg_df["Label"] = (
                agg_df["Period"].dt.year.astype(str) + " Q" +
                agg_df["Period"].dt.quarter.astype(str)
            )
        elif period == "Month":
            agg_df["Label"] = agg_df["Period"].dt.strftime("%b %Y")
        else:
            agg_df["Label"] = agg_df["Period"].dt.strftime("W%U %Y")

        agg_df["Growth (%)"] = agg_df["Value"].pct_change() * 100
        agg_df["Growth (%)"] = agg_df["Growth (%)"].round(2)

    except Exception as e:
        st.error(f"Could not process time data: {e}")
        return

    if len(agg_df) < 2:
        st.warning("Not enough data points to build a time series. "
                   "Try a finer period or check your date column.")
        return

    st.divider()

    total_periods = len(agg_df)
    total_val     = agg_df["Value"].sum()
    peak_label    = agg_df.loc[agg_df["Value"].idxmax(), "Label"]
    avg_growth    = agg_df["Growth (%)"].mean()

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, f"{total_periods}",       f"{period}s in Range")
    _metric_card(m2, f"{total_val:,.1f}",      f"Total {agg_fn}({value_col})")
    _metric_card(m3, peak_label,               "Peak Period")
    _metric_card(m4, f"{avg_growth:+.1f}%",    "Avg Period Growth")

    st.markdown("")

    st.markdown("### 📈 Trend Chart")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg_df["Label"],
        y=agg_df["Value"],
        mode="lines+markers",
        name=f"{agg_fn}({value_col})",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        marker=dict(size=6, color=PRIMARY_COLOR),
        hovertemplate="%{x}<br>Value: %{y:,.2f}<extra></extra>"
    ))

    if show_growth:
        fig.add_trace(go.Bar(
            x=agg_df["Label"],
            y=agg_df["Growth (%)"],
            name="Growth (%)",
            marker_color=[
                ACCENT_COLOR if v >= 0 else "#FF6B6B"
                for v in agg_df["Growth (%)"].fillna(0)
            ],
            opacity=0.5,
            yaxis="y2",
            hovertemplate="%{x}<br>Growth: %{y:+.2f}%<extra></extra>"
        ))
        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(color=ACCENT_COLOR),
                title=dict(text="Growth (%)", font=dict(color=ACCENT_COLOR))
            )
        )

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title=f"{agg_fn}({value_col}) by {period}",
        title_font_color=PRIMARY_COLOR,
        xaxis=dict(gridcolor="#333", tickangle=-45),
        yaxis=dict(gridcolor="#333", title=dict(text=f"{agg_fn}({value_col})")),
        legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#333"),
        margin=dict(t=60, b=80, l=60, r=60),
        height=450,
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.markdown("### 📋 Period Data Table")

    display = agg_df[["Label", "Value", "Growth (%)"]].copy()
    display.columns = [period, f"{agg_fn}({value_col})", "Growth (%)"]
    display = display.reset_index(drop=True)

    st.dataframe(display, width="stretch", height=300)

    st.divider()
    st.markdown("### 💾 Export")

    dc1, dc2 = st.columns(2)

    csv_bytes = display.to_csv(index=False).encode("utf-8")
    dc1.download_button(
        "⬇️ Download CSV", data=csv_bytes,
        file_name="time_intelligence.csv",
        mime="text/csv",
        width="stretch"
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        display.to_excel(writer, index=False, sheet_name="Time Intelligence")
    dc2.download_button(
        "⬇️ Download Excel", data=buffer.getvalue(),
        file_name="time_intelligence.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch"
    )


# ── Tab 4: Contribution & Pareto Analysis ───────────────────────────────────

def _contribution_pareto(df, numeric_cols, all_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Contribution Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        group_col = st.selectbox("📋 Group By", options=all_cols,
                                 index=0, key="cp_group")
    with c2:
        value_col = st.selectbox("🔢 Value Column", options=numeric_cols,
                                 index=0, key="cp_val")
    with c3:
        agg_fn = st.selectbox("∑ Aggregation",
                              options=["sum", "mean", "count",
                                       "median", "max", "min"],
                              index=0, key="cp_agg")

    max_groups = st.slider("Max Groups to Display", min_value=5,
                           max_value=50, value=15, step=1, key="cp_maxn")

    try:
        agg = (
            df.groupby(group_col)[value_col]
            .agg(agg_fn)
            .reset_index()
            .rename(columns={value_col: "Value"})
            .sort_values("Value", ascending=False)
            .reset_index(drop=True)
        )

        total_value = agg["Value"].sum()
        if total_value == 0:
            st.error("Total value is zero — cannot compute contribution %.")
            return

        agg["% of Total"] = (agg["Value"] / total_value * 100).round(2)
        agg["Cumulative %"] = agg["% of Total"].cumsum().round(2)

    except Exception as e:
        st.error(f"Could not compute contribution analysis: {e}")
        return

    if len(agg) < 2:
        st.warning("Need at least 2 groups to build a contribution analysis.")
        return

    pareto_cutoff_df = agg[agg["Cumulative %"] <= 80]
    groups_for_80pct = len(pareto_cutoff_df) + 1  
    groups_for_80pct = min(groups_for_80pct, len(agg))
    pct_of_groups = round(groups_for_80pct / len(agg) * 100, 1)

    display_agg = agg.head(max_groups).copy()

    st.divider()

    top_contributor   = agg.iloc[0][group_col]
    top_contribution  = agg.iloc[0]["% of Total"]

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, f"{len(agg)}",              "Total Groups")
    _metric_card(m2, str(top_contributor),        "Top Contributor")
    _metric_card(m3, f"{top_contribution:.1f}%",  "Top Group Share")
    _metric_card(m4, f"{groups_for_80pct} ({pct_of_groups}%)",
                 "Groups Driving 80% of Value")

    st.markdown("")
    st.markdown("### 🥧 Pareto Chart")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=display_agg[group_col].astype(str),
        y=display_agg["Value"],
        name=f"{agg_fn}({value_col})",
        marker_color=PRIMARY_COLOR,
        hovertemplate="%{x}<br>Value: %{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=display_agg[group_col].astype(str),
        y=display_agg["Cumulative %"],
        name="Cumulative %",
        mode="lines+markers",
        line=dict(color=ACCENT_COLOR, width=2.5),
        marker=dict(size=6, color=ACCENT_COLOR),
        yaxis="y2",
        hovertemplate="%{x}<br>Cumulative: %{y:.1f}%<extra></extra>"
    ))

    fig.add_shape(
        type="line", xref="paper", yref="y2",
        x0=0, x1=1, y0=80, y1=80,
        line=dict(color="#FF6B6B", width=1.5, dash="dash")
    )
    fig.add_annotation(
        xref="paper", yref="y2", x=1, y=80,
        text="80% line", showarrow=False,
        font=dict(color="#FF6B6B", size=11),
        xanchor="right", yanchor="bottom"
    )

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title=f"Pareto Analysis: {agg_fn}({value_col}) by {group_col}",
        title_font_color=PRIMARY_COLOR,
        xaxis=dict(gridcolor="#333", tickangle=-45),
        yaxis=dict(gridcolor="#333", title=dict(text=f"{agg_fn}({value_col})")),
        yaxis2=dict(
            overlaying="y", side="right",
            range=[0, 105], showgrid=False,
            tickfont=dict(color=ACCENT_COLOR),
            title=dict(text="Cumulative %", font=dict(color=ACCENT_COLOR))
        ),
        legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#333"),
        margin=dict(t=60, b=90, l=60, r=60),
        height=460,
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### 📋 Contribution Table")

    table = agg.copy()
    table.index = range(1, len(table) + 1)
    table.index.name = "Rank"
    st.dataframe(table, width="stretch", height=380)

    st.divider()
    st.markdown("### 💾 Export")

    dc1, dc2 = st.columns(2)

    csv_bytes = agg.to_csv(index=False).encode("utf-8")
    dc1.download_button(
        "⬇️ Download CSV", data=csv_bytes,
        file_name="contribution_pareto.csv",
        mime="text/csv",
        width="stretch"
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        agg.to_excel(writer, index=False, sheet_name="Contribution")
    dc2.download_button(
        "⬇️ Download Excel", data=buffer.getvalue(),
        file_name="contribution_pareto.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch"
    )