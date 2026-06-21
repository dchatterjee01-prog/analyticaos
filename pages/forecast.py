import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    .metric-value {{ font-size: 1.8rem; font-weight: 800; color: {ACCENT_COLOR}; }}
    .metric-label {{ font-size: 0.78rem; color: {TEXT_COLOR}; opacity: 0.75; margin-top: 0.2rem; }}
    </style>
    """, unsafe_allow_html=True)


def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── DATE DETECTION & FREQUENCY ───────────────────────────────────────────────
def _detect_date_cols(df: pd.DataFrame) -> list:
    date_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
                if parsed.notna().sum() > len(df) * 0.7:
                    date_cols.append(c)
            except Exception:
                pass
    return date_cols


def _infer_freq(date_series: pd.Series) -> tuple:
    diffs = date_series.sort_values().diff().dropna()
    if diffs.empty:
        return "D", 7, "daily"
    median_days = diffs.dt.total_seconds().median() / 86400

    if median_days <= 1.5: return "D", 7, "daily"
    elif median_days <= 9: return "W", 52, "weekly"
    elif median_days <= 45: return "ME", 12, "monthly"
    elif median_days <= 100: return "QE", 4, "quarterly"
    else: return "YE", 1, "yearly"


# ── MODELING ENGINES ─────────────────────────────────────────────────────────
def _holdout_mape(series: pd.Series, model_type: str, sp: int, order: tuple, s_order: tuple) -> float:
    n = len(series)
    if n < max(10, sp * 2 + 4): return None
    split = int(n * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    if len(test) == 0 or len(train) < sp * 2: return None

    try:
        if "Holt-Winters" in model_type:
            use_s = len(train) >= sp * 2 and sp > 1
            m = ExponentialSmoothing(train, trend="add", seasonal="add" if use_s else None, seasonal_periods=sp if use_s else None, initialization_method="estimated").fit()
            preds = m.forecast(len(test))
        else:
            m = SARIMAX(train, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            preds = m.forecast(len(test))
            
        actual, pred_vals = test.values, preds.values
        mask = actual != 0
        if mask.sum() == 0: return None
        return round(np.mean(np.abs((actual[mask] - pred_vals[mask]) / actual[mask])) * 100, 2)
    except Exception:
        return None


def _run_forecast(series: pd.Series, periods: int, model_type: str, sp: int, order: tuple, s_order: tuple):
    if "Holt-Winters" in model_type:
        use_s = len(series) >= sp * 2 and sp > 1
        m = ExponentialSmoothing(series, trend="add", seasonal="add" if use_s else None, seasonal_periods=sp if use_s else None, initialization_method="estimated").fit()
        fc = m.forecast(periods)
        resid_std = m.resid.dropna().std() if len(m.resid.dropna()) > 1 else series.std() * 0.1
        margin = 1.96 * resid_std * np.sqrt(np.arange(1, periods + 1))
        return fc.values, fc.values - margin, fc.values + margin
    else:
        m = SARIMAX(series, order=order, seasonal_order=s_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc_obj = m.get_forecast(steps=periods)
        mean = fc_obj.predicted_mean
        ci = fc_obj.conf_int(alpha=0.05)
        return mean.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── MAIN PAGE ────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("🔮 Forecasting Engine")
    st.caption("Phase 11.5 — Time Series Forecasting (Holt-Winters & ARIMA/SARIMA)")

    df: pd.DataFrame = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    date_cols = _detect_date_cols(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not date_cols or not num_cols:
        st.info("⚪ Requires at least one Date column and one Numeric column.")
        return

    st.markdown('<div class="section-header">Forecast Setup</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date column:", options=date_cols, key="fc_date_col")
    target_col = c2.selectbox("Target column:", options=num_cols, key="fc_target_col")
    horizon = c3.number_input("Forecast horizon (periods):", 1, 365, 30, key="fc_horizon")

    ts_df = df[[date_col, target_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce", format="mixed")
    ts_df = ts_df.dropna().sort_values(date_col).reset_index(drop=True)

    if len(ts_df) < 10:
        st.warning("⚠️ Need at least 10 valid data points for a meaningful forecast.")
        return

    freq, seasonal_periods, freq_label = _infer_freq(ts_df[date_col])
    series = ts_df.set_index(date_col)[target_col].resample(freq).mean().interpolate(method="linear")
    st.caption(f"Detected frequency: **{freq_label}** (Assumed Seasonality: {seasonal_periods} periods)")

    st.markdown('<div class="section-header">Model Selection</div>', unsafe_allow_html=True)
    model_type = st.radio("Algorithm:", ["Holt-Winters (Auto)", "ARIMA / SARIMA (Manual)"], horizontal=True)

    order = (1, 1, 1)
    s_order = (0, 0, 0, 0)

    if "ARIMA" in model_type:
        st.caption("⚙️ Configure Non-Seasonal (p, d, q) and Seasonal (P, D, Q) Parameters")
        ap1, ap2, ap3, sp1, sp2, sp3 = st.columns(6)
        p = ap1.number_input("p (AR)", 0, 5, 1)
        d = ap2.number_input("d (Diff)", 0, 2, 1)
        q = ap3.number_input("q (MA)", 0, 5, 1)
        order = (p, d, q)
        if seasonal_periods > 1:
            P = sp1.number_input("P", 0, 3, 1)
            D = sp2.number_input("D", 0, 2, 1)
            Q = sp3.number_input("Q", 0, 3, 1)
            s_order = (P, D, Q, seasonal_periods)
        else:
            sp1.info("No seasonality.")

    if st.button("🚀 Run Forecast", type="primary"):
        with st.spinner(f"Fitting {model_type}..."):
            try:
                forecast_vals, lower_ci, upper_ci = _run_forecast(series, horizon, model_type, seasonal_periods, order, s_order)
                mape = _holdout_mape(series, model_type, seasonal_periods, order, s_order)
            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")
                return

        future_index = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=freq)[1:]

        st.markdown('<div class="section-header">Forecast Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        _metric_card(m1, f"{len(series):,}", "Historical Points")
        _metric_card(m2, f"{horizon}", f"Periods ({freq_label})")
        _metric_card(m3, f"{forecast_vals[-1]:.2f}", "Final Forecast")
        _metric_card(m4, f"{mape}%" if mape else "N/A", "Holdout MAPE")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Historical", line=dict(color=PRIMARY_COLOR, width=2)))
        fig.add_trace(go.Scatter(x=future_index, y=forecast_vals, mode="lines", name="Forecast", line=dict(color=ACCENT_COLOR, width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(x=list(future_index) + list(future_index[::-1]), y=list(upper_ci) + list(lower_ci[::-1]), fill="toself", fillcolor=_hex_to_rgba(ACCENT_COLOR, 0.13), line=dict(color="rgba(0,0,0,0)"), name="95% Confidence Band"))
        fig.update_layout(
            title=f"{target_col} — {model_type} Forecast", paper_bgcolor=BACKGROUND_COLOR, plot_bgcolor=SURFACE_COLOR,
            font_color=TEXT_COLOR, margin=dict(t=50, b=50), height=460, legend=dict(bgcolor=SURFACE_COLOR)
        )
        st.plotly_chart(fig, width="stretch")

        out_df = pd.DataFrame({date_col: future_index, f"{target_col}_forecast": np.round(forecast_vals, 4), "lower_95": np.round(lower_ci, 4), "upper_95": np.round(upper_ci, 4)})
        with st.expander("📋 Forecast Data Table"):
            st.dataframe(out_df.astype(str), width="stretch", hide_index=True)
            st.download_button("⬇️ Download Forecast (CSV)", data=out_df.to_csv(index=False).encode("utf-8"), file_name=f"forecast_{target_col}.csv", mime="text/csv")