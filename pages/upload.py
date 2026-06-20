# pages/upload.py

import streamlit as st
import pandas as pd
from config.settings import ACCENT_COLOR, PRIMARY_COLOR, TEXT_COLOR

def render():
    st.markdown("## 📁 Upload Your Dataset")
    st.markdown(
        "Supported formats: **CSV** and **Excel (.xlsx)**. "
        "Your data never leaves your machine."
    )

    st.divider()

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        label="Drop your file here or click to browse",
        type=["csv", "xlsx"],
        help="Maximum file size: 200 MB"
    )

    if uploaded_file is None:
        _show_placeholder()
        return

    # ── Load file ────────────────────────────────────────────────────────────
    with st.spinner("Reading file..."):
        df, error = _load_file(uploaded_file)

    if error:
        st.error(f"Could not read file: {error}")
        return

    # ── Sanitize mixed-type columns ──────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)

    # ── Store in session state ───────────────────────────────────────────────
    st.session_state["df"]          = df
    st.session_state["df_original"] = df.copy()
    st.session_state["filename"]    = uploaded_file.name

    # ── Success banner ───────────────────────────────────────────────────────
    st.success(f"✅ **{uploaded_file.name}** loaded successfully.")

    # ── Dataset summary cards ────────────────────────────────────────────────
    _show_summary_cards(df)

    st.divider()

    # ── Data preview ─────────────────────────────────────────────────────────
    st.markdown("### 🔍 Data Preview")
    n_rows = st.slider("Rows to preview", min_value=5, max_value=50,
                       value=10, step=5)
    st.dataframe(df.head(n_rows), width='stretch')

    st.divider()

    # ── Column info table ────────────────────────────────────────────────────
    st.markdown("### 🗂️ Column Information")
    col_info = _build_column_info(df)
    st.dataframe(col_info, width='stretch')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_file(uploaded_file):
    """Load CSV or Excel into a DataFrame. Returns (df, error_string)."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            return None, "Unsupported file format."
        return df, None
    except Exception as e:
        return None, str(e)


def _show_summary_cards(df):
    """Display 4 metric cards: rows, columns, missing values, duplicates."""
    total_cells   = df.shape[0] * df.shape[1]
    missing_vals  = df.isnull().sum().sum()
    missing_pct   = round((missing_vals / total_cells) * 100, 1) if total_cells else 0
    duplicate_rows = df.duplicated().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Rows",         f"{df.shape[0]:,}")
    c2.metric("📋 Columns",      f"{df.shape[1]:,}")
    c3.metric("❓ Missing Values", f"{missing_vals:,}",
              delta=f"{missing_pct}% of cells",
              delta_color="inverse")
    c4.metric("♻️ Duplicate Rows", f"{duplicate_rows:,}",
              delta_color="inverse")


def _build_column_info(df):
    """Return a summary DataFrame with column name, dtype, nulls, and sample."""
    info = []
    for col in df.columns:
        info.append({
            "Column":        col,
            "Data Type":     str(df[col].dtype),
            "Non-Null":      df[col].notna().sum(),
            "Null Count":    df[col].isnull().sum(),
            "Unique Values": df[col].nunique(),
            "Sample Value":  str(df[col].dropna().iloc[0])
                             if df[col].notna().any() else "—"
        })
    return pd.DataFrame(info)


def _show_placeholder():
    """Shown before any file is uploaded."""
    st.markdown("""
    <div style="
        border: 2px dashed #6C63FF44;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        color: #E8E8F099;
        margin-top: 1rem;
    ">
        <div style="font-size: 2.5rem;">📂</div>
        <div style="font-size: 1rem; margin-top: 0.5rem;">
            No dataset loaded yet.<br>
            Upload a CSV or Excel file above to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)