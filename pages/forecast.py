import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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


# ── DATE COLUMN DETECTION (mirrors pages/eda.py pattern, with pandas 3.0 fix) ─
def _detect_date_cols(df: pd.DataFrame) -> list:
    date_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            # Handles datetime64[ns], datetime64[us], etc. — pandas 3.0 changed
            # the default resolution from [ns] to [us], breaking strict
            # dtype == "datetime64[ns]" string comparisons (see Phase 9 note).
            date_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
                if parsed.notna().sum() > len(df) * 0.7:
                    date_cols.append(c)
            except Exception:
                pass
    return date_cols


# ── INFER FREQUENCY FROM DATE SPACING ────────────────────────────────────────
def _infer_freq(date_series: pd.Series) -> tuple:
    """Returns (pandas_freq_str, seasonal_periods, label) based on median gap."""
    diffs = date_series.sort_values().diff().dropna()
    if diffs.empty:
        return "D", 7, "daily"
    median_days = diffs.dt.total_seconds().median() / 86400

    if median_days <= 1.5:
        return "D", 7, "daily"
    elif median_days <= 9:
        return "W", 52, "weekly"
    elif median_days <= 45:
        return "ME", 12, "monthly"
    elif median_days <= 100:
        return "QE", 4, "quarterly"
    else:
        return "YE", 1, "yearly"


# ── HOLDOUT MAPE ──────────────────────────────────────────────────────────────
def _holdout_mape(series: pd.Series, seasonal_periods: int) -> float:
    """Fits on first 80%, predicts remaining 20%, returns MAPE %. None if not enough data."""
    n = len(series)
    if n < max(10, seasonal_periods * 2 + 4):
        return None

    split = int(n * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    if len(test) == 0 or len(train) < seasonal_periods * 2:
        return None

    try:
        use_seasonal = len(train) >= seasonal_periods * 2
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated",
        ).fit()
        preds = model.forecast(len(test))
        actual = test.values
        pred_vals = preds.values
        mask = actual != 0
        if mask.sum() == 0:
            return None
        mape = np.mean(np.abs((actual[mask] - pred_vals[mask]) / actual[mask])) * 100
        return round(mape, 2)
    except Exception:
        return None


# ── FORECAST ──────────────────────────────────────────────────────────────────
def _run_forecast(series: pd.Series, periods: int, seasonal_periods: int):
    """Fits ExponentialSmoothing on full series, forecasts `periods` ahead.
    Returns (forecast_values, lower_ci, upper_ci)."""
    n = len(series)
    use_seasonal = n >= seasonal_periods * 2

    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    ).fit()

    forecast = model.forecast(periods)

    # Simple confidence band: residual std-based, widening with horizon
    resid = model.resid.dropna()
    resid_std = resid.std() if len(resid) > 1 else series.std() * 0.1
    horizon_factor = np.sqrt(np.arange(1, periods + 1))
    margin = 1.96 * resid_std * horizon_factor

    lower = forecast.values - margin
    upper = forecast.values + margin

    return forecast.values, lower, upper


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Converts '#RRGGBB' to 'rgba(r,g,b,a)' since Plotly's fillcolor
    does not accept the CSS-style '#RRGGBBAA' 8-digit hex format."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("🔮 Forecasting Engine")
    st.caption("Phase 11.5 — Time Series Forecasting")

    df: pd.DataFrame = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    # ── Date column detection ──
    date_cols = _detect_date_cols(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not date_cols:
        st.info(
            "⚪ No date columns detected. "
            "Use Data Cleaning → Data Type Fixer to convert a column to datetime first."
        )
        return

    if not num_cols:
        st.info("No numeric columns available to forecast.")
        return

    st.success(f"✅ Date columns detected: {', '.join(date_cols)}")

    # ── Selectors ──
    st.markdown('<div class="section-header">Forecast Setup</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date column:", options=date_cols, key="fc_date_col")
    target_col = c2.selectbox("Target column:", options=num_cols, key="fc_target_col")
    horizon = c3.number_input(
        "Forecast horizon (periods):",
        min_value=1, max_value=365, value=30, key="fc_horizon"
    )

    # ── Prepare series ──
    ts_df = df[[date_col, target_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce", format="mixed")
    ts_df = ts_df.dropna().sort_values(date_col).reset_index(drop=True)

    if len(ts_df) < 10:
        st.warning(
            f"⚠️ Only {len(ts_df)} valid data points after cleaning — "
            "need at least 10 for a meaningful forecast."
        )
        return

    freq, seasonal_periods, freq_label = _infer_freq(ts_df[date_col])
    series = ts_df.set_index(date_col)[target_col]

    # Resample to inferred frequency to ensure even spacing (required by statsmodels)
    series = series.resample(freq).mean().interpolate(method="linear")

    st.caption(f"Detected frequency: **{freq_label}** ({len(series)} points after resampling)")

    # ── Run forecast ──
    with st.spinner("Fitting model and forecasting..."):
        try:
            forecast_vals, lower_ci, upper_ci = _run_forecast(series, horizon, seasonal_periods)
            mape = _holdout_mape(series, seasonal_periods)
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")
            st.info(
                "Try a different target column, a longer date range, "
                "or a smaller forecast horizon."
            )
            return

    future_index = pd.date_range(
        start=series.index[-1], periods=horizon + 1, freq=freq
    )[1:]

    # ── Metrics row ──
    st.markdown('<div class="section-header">Forecast Summary</div>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, f"{len(series):,}", "Historical Points")
    _metric_card(m2, f"{horizon}", f"Forecast Periods ({freq_label})")
    _metric_card(m3, f"{forecast_vals[-1]:.2f}", "Final Forecast Value")
    if mape is not None:
        _metric_card(m4, f"{mape}%", "Holdout MAPE")
    else:
        _metric_card(m4, "N/A", "Holdout MAPE (insufficient data)")

    # ── Chart ──
    st.markdown('<div class="section-header">Historical + Forecast</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name="Historical",
        line=dict(color=PRIMARY_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=future_index, y=forecast_vals,
        mode="lines", name="Forecast",
        line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=list(future_index) + list(future_index[::-1]),
        y=list(upper_ci) + list(lower_ci[::-1]),
        fill="toself",
        fillcolor=_hex_to_rgba(ACCENT_COLOR, 0.13),
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
        showlegend=True,
    ))
    fig.update_layout(
        title=f"{target_col} — Forecast ({freq_label})",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(title=date_col, gridcolor="#2a2a45"),
        yaxis=dict(title=target_col, gridcolor="#2a2a45"),
        legend=dict(bgcolor=SURFACE_COLOR, bordercolor="#2a2a45"),
        margin=dict(t=50, b=50),
        height=460,
    )
    st.plotly_chart(fig, width="stretch")

    # ── Forecast table + download ──
    with st.expander("📋 Forecast Data Table"):
        out_df = pd.DataFrame({
            date_col: future_index,
            f"{target_col}_forecast": forecast_vals.round(4),
            "lower_95": lower_ci.round(4),
            "upper_95": upper_ci.round(4),
        })
        st.dataframe(out_df.astype(str), width="stretch", hide_index=True)

        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecast (CSV)",
            data=csv,
            file_name=f"forecast_{target_col}.csv",
            mime="text/csv",
            key="dl_forecast",
        )

    # ── Methodology note ──
    method_used = "Holt-Winters (additive trend + seasonality)" if len(series) >= seasonal_periods * 2 else "Holt's linear trend (insufficient data for seasonality)"
    st.caption(f"Method: {method_used} · Seasonal periods assumed: {seasonal_periods}")