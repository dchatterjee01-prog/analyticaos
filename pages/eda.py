import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
    .col-badge-num {{
        background: #1a2a3a;
        color: #64B5F6;
        border: 1px solid #64B5F6;
        border-radius: 6px;
        padding: 0.1rem 0.5rem;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    .col-badge-cat {{
        background: #2a1a3a;
        color: #CE93D8;
        border: 1px solid #CE93D8;
        border-radius: 6px;
        padding: 0.1rem 0.5rem;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── METRIC CARD ───────────────────────────────────────────────────────────────
def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── STATISTICAL SUMMARY ───────────────────────────────────────────────────────
def _stat_summary(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return pd.DataFrame()

    desc = df[num_cols].describe().T.reset_index()
    desc.columns = ["Column","Count","Mean","Std","Min",
                    "25%","50% (Median)","75%","Max"]
    desc = desc.round(4)
    return desc


# ── HISTOGRAM ─────────────────────────────────────────────────────────────────
def _histogram(df: pd.DataFrame, col: str):
    data = df[col].dropna()
    fig  = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=40,
        marker_color=PRIMARY_COLOR,
        opacity=0.85,
        name=col,
    ))
    fig.update_layout(
        title=f"Distribution — {col}",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(gridcolor="#2a2a45"),
        yaxis=dict(title="Frequency", gridcolor="#2a2a45"),
        margin=dict(t=50, b=40),
        height=350,
    )
    return fig


# ── BOX PLOT ──────────────────────────────────────────────────────────────────
def _boxplot(df: pd.DataFrame, col: str):
    data = df[col].dropna()
    fig  = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        name=col,
        marker_color=ACCENT_COLOR,
        boxmean=True,
    ))
    fig.update_layout(
        title=f"Box Plot — {col}",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        yaxis=dict(gridcolor="#2a2a45"),
        margin=dict(t=50, b=40),
        height=350,
    )
    return fig


# ── VALUE COUNTS BAR ──────────────────────────────────────────────────────────
def _value_counts_bar(df: pd.DataFrame, col: str, top_n: int = 15):
    vc   = df[col].value_counts().head(top_n)
    fig  = go.Figure(go.Bar(
        x=vc.index.astype(str),
        y=vc.values,
        marker_color=PRIMARY_COLOR,
        opacity=0.85,
    ))
    fig.update_layout(
        title=f"Top {top_n} Value Counts — {col}",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(tickangle=-35, gridcolor="#2a2a45"),
        yaxis=dict(title="Count", gridcolor="#2a2a45"),
        margin=dict(t=50, b=80),
        height=350,
    )
    return fig


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("🔬 EDA Engine")
    st.caption("Phase 3 — Exploratory Data Analysis")

    df: pd.DataFrame = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── TOP METRICS ──
    st.markdown('<div class="section-header">Dataset Snapshot</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    _metric_card(c1, f"{len(df):,}",        "Rows")
    _metric_card(c2, f"{df.shape[1]}",      "Columns")
    _metric_card(c3, f"{len(num_cols)}",    "Numeric Cols")
    _metric_card(c4, f"{len(cat_cols)}",    "Text Cols")
    _metric_card(c5, f"{df.isnull().sum().sum():,}", "Missing Cells")

    # ── STATISTICAL SUMMARY ──
    st.markdown('<div class="section-header">Statistical Summary</div>',
                unsafe_allow_html=True)
    summary = _stat_summary(df)
    if summary.empty:
        st.info("No numeric columns found.")
    else:
        st.dataframe(summary.astype(str), width='stretch', hide_index=True)

    # ── COLUMN PROFILER ──
    st.markdown("---")
    st.markdown('<div class="section-header">Column Profiler</div>',
                unsafe_allow_html=True)

    selected_col = st.selectbox(
        "Select a column to profile:",
        options=df.columns.tolist(),
        key="eda_col"
    )

    col_data  = df[selected_col]
    col_dtype = str(col_data.dtype)
    is_num    = col_data.dtype.kind in "iufcb"

    # ── Column metrics ──
    p1, p2, p3, p4 = st.columns(4)
    _metric_card(p1, col_dtype,                        "Data Type")
    _metric_card(p2, f"{col_data.nunique():,}",        "Unique Values")
    _metric_card(p3, f"{col_data.isnull().sum():,}",   "Missing")
    _metric_card(p4,
        f"{round(col_data.isnull().sum()/len(df)*100,1)}%",
        "Missing %"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if is_num:
        # Numeric — histogram + boxplot side by side
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(_histogram(df, selected_col), width='stretch')
        with ch2:
            st.plotly_chart(_boxplot(df, selected_col), width='stretch')

        # Extra numeric stats
        st.markdown('<div class="section-header">Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        _metric_card(s1, f"{col_data.mean():.4f}",   "Mean")
        _metric_card(s2, f"{col_data.median():.4f}", "Median")
        _metric_card(s3, f"{col_data.std():.4f}",    "Std Dev")
        _metric_card(s4, f"{col_data.skew():.4f}",   "Skewness")
        _metric_card(s5, f"{col_data.kurt():.4f}",   "Kurtosis")
        _metric_card(s6, f"{col_data.max() - col_data.min():.4f}", "Range")

    else:
        # Categorical — value counts bar
        st.plotly_chart(
            _value_counts_bar(df, selected_col),
            width='stretch'
        )
        with st.expander("📋 Full Value Counts Table"):
            vc_df = (df[selected_col]
                     .value_counts()
                     .reset_index())
            vc_df.columns = ["Value", "Count"]
            vc_df["Percentage"] = (
                vc_df["Count"] / len(df) * 100
            ).round(2)
            st.dataframe(vc_df.astype(str), width='stretch', hide_index=True)
            # ════════════════════════════════════════════════════════════════
    # SECTION 2 — CORRELATION ANALYSIS
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🔗 Correlation Analysis</div>',
                unsafe_allow_html=True)

    num_cols_corr = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols_corr) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        # ── Method selector ──
        corr_method = st.selectbox(
            "Correlation method:",
            options=["pearson", "spearman", "kendall"],
            key="corr_method"
        )

        corr_matrix = df[num_cols_corr].corr(method=corr_method).round(3)

        # ── Heatmap ──
        st.markdown('<div class="section-header">Correlation Heatmap</div>',
                    unsafe_allow_html=True)

        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            showscale=True,
        ))
        fig_corr.update_layout(
            title=f"{corr_method.capitalize()} Correlation Matrix",
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=SURFACE_COLOR,
            font_color=TEXT_COLOR,
            height=max(400, len(num_cols_corr) * 50 + 100),
            margin=dict(t=50, b=80, l=100, r=40),
            xaxis=dict(tickangle=-35),
        )
        st.plotly_chart(fig_corr, width='stretch')

        # ── Top correlated pairs ──
        st.markdown('<div class="section-header">Top Correlated Pairs</div>',
                    unsafe_allow_html=True)

        pairs = []
        for i in range(len(num_cols_corr)):
            for j in range(i+1, len(num_cols_corr)):
                c1 = num_cols_corr[i]
                c2 = num_cols_corr[j]
                r  = corr_matrix.loc[c1, c2]
                pairs.append({
                    "Column A"    : c1,
                    "Column B"    : c2,
                    "Correlation" : round(r, 4),
                    "Strength"    : (
                        "🔴 Very Strong" if abs(r) >= 0.9 else
                        "🟠 Strong"      if abs(r) >= 0.7 else
                        "🟡 Moderate"    if abs(r) >= 0.5 else
                        "🟢 Weak"        if abs(r) >= 0.3 else
                        "⚪ Negligible"
                    ),
                    "Direction"   : "⬆️ Positive" if r > 0 else "⬇️ Negative"
                })

        pairs_df = (pd.DataFrame(pairs)
                    .sort_values("Correlation",
                                 key=abs,
                                 ascending=False)
                    .reset_index(drop=True))
        st.dataframe(pairs_df.astype(str), width='stretch', hide_index=True)

        # ── Multicollinearity warning ──
        high_corr = pairs_df[abs(pairs_df["Correlation"]) >= 0.9]
        if not high_corr.empty:
            st.warning(
                f"⚠️ Multicollinearity detected — "
                f"{len(high_corr)} pairs have correlation ≥ 0.9. "
                f"Consider dropping one column from each pair before ML."
            )
        else:
            st.success("✅ No multicollinearity detected (no pair ≥ 0.9).")

        # ── Scatter plot for selected pair ──
        st.markdown('<div class="section-header">Scatter Plot — Column Pair</div>',
                    unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        x_col = sc1.selectbox("X axis:", options=num_cols_corr, key="scatter_x")
        y_col = sc2.selectbox("Y axis:", options=num_cols_corr,
                              index=min(1, len(num_cols_corr)-1),
                              key="scatter_y")

        fig_scatter = go.Figure(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(
                color=PRIMARY_COLOR,
                size=5,
                opacity=0.6,
            ),
            name=f"{x_col} vs {y_col}",
        ))
        fig_scatter.update_layout(
            title=f"{x_col} vs {y_col}",
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=SURFACE_COLOR,
            font_color=TEXT_COLOR,
            xaxis=dict(title=x_col, gridcolor="#2a2a45"),
            yaxis=dict(title=y_col, gridcolor="#2a2a45"),
            margin=dict(t=50, b=50),
            height=400,
        )
        st.plotly_chart(fig_scatter, width='stretch')
        # ════════════════════════════════════════════════════════════════
    # SECTION 3 — OUTLIER DETECTION
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🚨 Outlier Detection</div>',
                unsafe_allow_html=True)

    num_cols_out = df.select_dtypes(include="number").columns.tolist()

    if not num_cols_out:
        st.info("No numeric columns found for outlier detection.")
    else:
        # ── Method selector ──
        out_method = st.selectbox(
            "Detection method:",
            options=["IQR (Interquartile Range)", "Z-Score"],
            key="out_method"
        )

        out_col = st.selectbox(
            "Select column:",
            options=num_cols_out,
            key="out_col"
        )

        col_data_out = df[out_col].dropna()

        # ── Detect outliers ──
        if out_method == "IQR (Interquartile Range)":
            Q1  = col_data_out.quantile(0.25)
            Q3  = col_data_out.quantile(0.75)
            IQR = Q3 - Q1
            lower  = Q1 - 1.5 * IQR
            upper  = Q3 + 1.5 * IQR
            outlier_mask = (
                (df[out_col] < lower) |
                (df[out_col] > upper)
            )
            method_desc = (
                f"IQR={IQR:.4f} | "
                f"Lower fence={lower:.4f} | "
                f"Upper fence={upper:.4f}"
            )

        else:  # Z-Score
            mean    = col_data_out.mean()
            std     = col_data_out.std()
            z_scores = (df[out_col] - mean) / std
            outlier_mask = abs(z_scores) > 3
            method_desc = (
                f"Mean={mean:.4f} | "
                f"Std={std:.4f} | "
                f"Threshold=±3σ"
            )

        outlier_count = outlier_mask.sum()
        outlier_pct   = round(outlier_count / len(df) * 100, 2)

        # ── Metrics ──
        st.markdown("<br>", unsafe_allow_html=True)
        o1, o2, o3 = st.columns(3)
        _metric_card(o1, f"{len(df):,}",        "Total Rows")
        _metric_card(o2, f"{outlier_count:,}",  "Outliers Found")
        _metric_card(o3, f"{outlier_pct}%",     "Outlier %")

        st.markdown(f"<br><small>{method_desc}</small>",
                    unsafe_allow_html=True)

        if outlier_count == 0:
            st.success(f"✅ No outliers detected in '{out_col}'.")
        else:
            st.warning(
                f"⚠️ {outlier_count} outliers found in '{out_col}' "
                f"using {out_method}."
            )

            # ── Box plot with outliers highlighted ──
            st.markdown(
                '<div class="section-header">Outlier Visualization</div>',
                unsafe_allow_html=True)

            normal_data  = df[out_col][~outlier_mask].dropna()
            outlier_data = df[out_col][outlier_mask].dropna()

            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(
                x=list(range(len(df))),
                y=df[out_col],
                mode="markers",
                marker=dict(color=ACCENT_COLOR, size=4, opacity=0.5),
                name="Normal",
            ))
            fig_out.add_trace(go.Scatter(
                x=df[out_col][outlier_mask].index.tolist(),
                y=outlier_data,
                mode="markers",
                marker=dict(color="#FF4B4B", size=8, symbol="x"),
                name="Outlier",
            ))
            fig_out.update_layout(
                title=f"Outliers in '{out_col}' — red X marks",
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=SURFACE_COLOR,
                font_color=TEXT_COLOR,
                xaxis=dict(title="Row Index", gridcolor="#2a2a45"),
                yaxis=dict(title=out_col, gridcolor="#2a2a45"),
                margin=dict(t=50, b=50),
                height=380,
            )
            st.plotly_chart(fig_out, width='stretch')

            # ── Box plot ──
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=df[out_col],
                name=out_col,
                marker_color=PRIMARY_COLOR,
                boxmean=True,
                boxpoints="outliers",
                marker=dict(
                    outliercolor="#FF4B4B",
                    size=6,
                ),
            ))
            fig_box.update_layout(
                title=f"Box Plot with Outliers — '{out_col}'",
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=SURFACE_COLOR,
                font_color=TEXT_COLOR,
                yaxis=dict(gridcolor="#2a2a45"),
                margin=dict(t=50, b=40),
                height=350,
            )
            st.plotly_chart(fig_box, width='stretch')

            # ── Preview outlier rows ──
            with st.expander(
                f"👁️ Preview {outlier_count} outlier rows"
            ):
                st.dataframe(
                    df[outlier_mask].astype(str),
                    width='stretch',
                    hide_index=False
                    
    
                )

            # ── Remove outliers ──
            st.markdown(
                '<div class="section-header">Remove Outliers</div>',
                unsafe_allow_html=True)

            if st.button(
                f"🗑️ Remove {outlier_count} outlier rows",
                key="remove_outliers"
            ):
                clean_df      = df[~outlier_mask].reset_index(drop=True)
                removed       = len(df) - len(clean_df)
                st.session_state["df"] = clean_df
                st.success(
                    f"✅ Removed {removed} outlier rows. "
                    f"Dataset now has {len(clean_df):,} rows."
                )

                # Download
                out_csv = clean_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Outlier-Free Dataset (CSV)",
                    data=out_csv,
                    file_name="outlier_free_dataset.csv",
                    mime="text/csv",
                    key="dl_outlier_free"
                )
                # ════════════════════════════════════════════════════════════════
    # SECTION 4 — TIME SERIES DETECTOR
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">📅 Time Series Detector</div>',
                unsafe_allow_html=True)

    # ── Auto detect date columns ──
    date_cols = []
    for c in df.columns:
        if df[c].dtype == "datetime64[ns]":
            date_cols.append(c)
        elif df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
                if parsed.notna().sum() > len(df) * 0.7:
                    date_cols.append(c)
            except Exception:
                pass

    if not date_cols:
        st.info(
            "⚪ No date columns detected. "
            "Use Data Type Fixer to convert a column to datetime first."
        )
    else:
        st.success(
            f"✅ Date columns detected: {', '.join(date_cols)}"
        )

        # ── Selectors ──
        ts1, ts2 = st.columns(2)
        date_col = ts1.selectbox(
            "Select date column:",
            options=date_cols,
            key="ts_date_col"
        )

        num_cols_ts = df.select_dtypes(include="number").columns.tolist()
        if not num_cols_ts:
            st.info("No numeric columns to plot.")
        else:
            value_col = ts2.selectbox(
                "Select value column:",
                options=num_cols_ts,
                key="ts_value_col"
            )

            # ── Prepare data ──
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df = ts_df.dropna().sort_values(date_col)

            # ── Rolling window ──
            roll_window = st.slider(
                "Rolling average window (days):",
                min_value=2,
                max_value=min(90, len(ts_df) // 2),
                value=min(7, len(ts_df) // 2),
                key="ts_roll"
            )
            ts_df["Rolling Avg"] = (
                ts_df[value_col]
                .rolling(window=roll_window)
                .mean()
            )

            # ── Time series plot ──
            st.markdown(
                '<div class="section-header">Time Series Plot</div>',
                unsafe_allow_html=True)

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df[value_col],
                mode="lines",
                name=value_col,
                line=dict(color=PRIMARY_COLOR, width=1.5),
                opacity=0.7,
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_df[date_col],
                y=ts_df["Rolling Avg"],
                mode="lines",
                name=f"{roll_window}-period Rolling Avg",
                line=dict(color=ACCENT_COLOR, width=2.5),
            ))
            fig_ts.update_layout(
                title=f"{value_col} over Time",
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=SURFACE_COLOR,
                font_color=TEXT_COLOR,
                xaxis=dict(title=date_col, gridcolor="#2a2a45"),
                yaxis=dict(title=value_col, gridcolor="#2a2a45"),
                legend=dict(
                    bgcolor=SURFACE_COLOR,
                    bordercolor="#2a2a45"
                ),
                margin=dict(t=50, b=50),
                height=420,
            )
            st.plotly_chart(fig_ts, width='stretch')

            # ── Time series metrics ──
            st.markdown(
                '<div class="section-header">Time Series Summary</div>',
                unsafe_allow_html=True)

            t1, t2, t3, t4, t5 = st.columns(5)
            _metric_card(t1,
                str(ts_df[date_col].min().date()),
                "Start Date")
            _metric_card(t2,
                str(ts_df[date_col].max().date()),
                "End Date")
            _metric_card(t3,
                f"{len(ts_df):,}",
                "Data Points")
            _metric_card(t4,
                f"{ts_df[value_col].max():.2f}",
                "Peak Value")
            _metric_card(t5,
                f"{ts_df[value_col].min():.2f}",
                "Lowest Value")

            # ── Trend direction ──
            st.markdown("<br>", unsafe_allow_html=True)
            first_val = ts_df[value_col].iloc[0]
            last_val  = ts_df[value_col].iloc[-1]
            change    = last_val - first_val
            change_pct = round(change / first_val * 100, 2)

            if change > 0:
                st.markdown(
                    f'<span class="clean-badge">'
                    f'📈 Upward Trend — '
                    f'+{change_pct}% from start to end'
                    f'</span>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="warn-badge">'
                    f'📉 Downward Trend — '
                    f'{change_pct}% from start to end'
                    f'</span>',
                    unsafe_allow_html=True)

            # ── Monthly aggregation ──
            st.markdown(
                '<div class="section-header">Monthly Aggregation</div>',
                unsafe_allow_html=True)

            try:
                monthly = (
                    ts_df.set_index(date_col)[value_col]
                    .resample("ME")
                    .agg(["mean", "min", "max"])
                    .reset_index()
                )
                monthly.columns = [
                    "Month", "Mean", "Min", "Max"
                ]
                monthly["Month"] = monthly["Month"].dt.strftime("%Y-%m")
                monthly = monthly.round(2)

                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Bar(
                    x=monthly["Month"],
                    y=monthly["Mean"],
                    marker_color=PRIMARY_COLOR,
                    name="Monthly Mean",
                    opacity=0.85,
                ))
                fig_monthly.update_layout(
                    title=f"Monthly Mean — {value_col}",
                    paper_bgcolor=BACKGROUND_COLOR,
                    plot_bgcolor=SURFACE_COLOR,
                    font_color=TEXT_COLOR,
                    xaxis=dict(
                        tickangle=-45,
                        gridcolor="#2a2a45"
                    ),
                    yaxis=dict(gridcolor="#2a2a45"),
                    margin=dict(t=50, b=80),
                    height=380,
                )
                st.plotly_chart(fig_monthly, width='stretch')

                with st.expander("📋 Monthly Data Table"):
                    st.dataframe(
                       monthly.astype(str),
                        width='stretch',
                        hide_index=True
                    )

            except Exception as e:
                st.warning(f"Monthly aggregation skipped: {e}")