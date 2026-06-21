import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR,
    TEXT_COLOR, BACKGROUND_COLOR,
    SUCCESS_COLOR, SUCCESS_BG, WARNING_COLOR, WARNING_BG
)

# ── PAGE STYLES ──────────────────────────────────────────────────────────────
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
        background: {SUCCESS_BG};
        color: {SUCCESS_COLOR};
        border: 1px solid {SUCCESS_COLOR};
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }}
    .warn-badge {{
        background: {WARNING_BG};
        color: {WARNING_COLOR};
        border: 1px solid {WARNING_COLOR};
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── METRIC CARD HELPER ───────────────────────────────────────────────────────
def _metric_card(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ── MISSING VALUE TABLE ──────────────────────────────────────────────────────
def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct   = (missing_count / len(df) * 100).round(2)
    dtype_map     = df.dtypes.astype(str)

    summary = pd.DataFrame({
        "Column"         : missing_count.index,
        "Missing Count"  : missing_count.values,
        "Missing %"      : missing_pct.values,
        "Data Type"      : dtype_map.values,
        "Non-Null Count" : df.notnull().sum().values,
    })
    summary = summary[summary["Missing Count"] > 0].reset_index(drop=True)
    summary = summary.sort_values("Missing %", ascending=False).reset_index(drop=True)
    return summary


# ── BAR CHART ────────────────────────────────────────────────────────────────
def _bar_chart(summary: pd.DataFrame):
    colors = [
        "#FF4B4B" if p >= 50 else
        "#FFB347" if p >= 20 else
        ACCENT_COLOR
        for p in summary["Missing %"]
    ]
    fig = go.Figure(go.Bar(
        x=summary["Column"],
        y=summary["Missing %"],
        marker_color=colors,
        text=[f"{p}%" for p in summary["Missing %"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Missing Values by Column (%)",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        xaxis=dict(tickangle=-35, gridcolor="#2a2a45"),
        yaxis=dict(title="Missing %", range=[0, 110], gridcolor="#2a2a45"),
        margin=dict(t=50, b=80),
        height=380,
    )
    return fig


# ── HEATMAP ──────────────────────────────────────────────────────────────────
def _heatmap(df: pd.DataFrame, cols_with_missing: list, sample: int = 200):
    sample_df = df[cols_with_missing].head(sample)
    z = sample_df.isnull().astype(int).T.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(min(sample, len(df)))),
        y=cols_with_missing,
        colorscale=[[0, SURFACE_COLOR], [1, "#FF4B4B"]],
        showscale=False,
    ))
    fig.update_layout(
        title=f"Missingness Pattern — first {min(sample, len(df))} rows (red = missing)",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font_color=TEXT_COLOR,
        height=max(250, len(cols_with_missing) * 35 + 80),
        margin=dict(t=50, b=40, l=130),
        xaxis=dict(showticklabels=False),
    )
    return fig


# ── MAIN PAGE ────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("🧹 Data Cleaning Engine")
    st.caption("Phase 2 — Missing Value Analyzer")

    df: pd.DataFrame = st.session_state.get("df")

    # Guard
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    # --- Stage A / Step Group 1 / Step 1: Reference dataset for drift detection ---
    with st.expander("📊 Drift Detection Setup (optional)", expanded=False):
        st.caption(
            "Upload a reference dataset (e.g. last month's data, or a known-good "
            "snapshot) to enable drift detection against your current dataset."
        )

        ref_file = st.file_uploader(
            "Reference dataset (.csv)",
            type=["csv"],
            key="drift_reference_uploader",
        )

        if ref_file is not None:
            try:
                ref_df = pd.read_csv(ref_file)
                st.session_state["drift_reference_df"] = ref_df
                st.success(
                    f"Reference dataset loaded: {ref_df.shape[0]:,} rows × "
                    f"{ref_df.shape[1]} columns"
                )
            except Exception as e:
                st.error(f"Could not read reference dataset: {e}")

        if "drift_reference_df" in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Reference rows",
                    f"{st.session_state['drift_reference_df'].shape[0]:,}",
                )
            with col2:
                if st.button("Clear reference dataset", width="stretch"):
                    del st.session_state["drift_reference_df"]
                    st.rerun()
        else:
            st.info("No reference dataset set. Drift detection will be unavailable until one is uploaded.")

    # --- Stage A / Step Group 1 / Step 3: Show drift + trust results ---
    agent_report = st.session_state.get("agent_report")
    dq_result = (agent_report or {}).get("results", {}).get("DataQualityAgent")

    if dq_result is not None:
        drift = dq_result.artifacts.get("drift", {})
        trust = dq_result.artifacts.get("trust", {})

        if drift.get("drift_available") or trust.get("trust_available"):
            st.markdown("##### Drift & Trust Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                if drift.get("drift_available"):
                    n_drifted = drift.get("n_drifted_columns", 0)
                    n_checked = drift.get("n_columns_checked", 0)
                    st.metric("Columns drifted", f"{n_drifted} / {n_checked}")
                else:
                    st.metric("Columns drifted", "—")

            with col2:
                if drift.get("drift_available") and drift.get("drift_share") is not None:
                    st.metric("Drift share", f"{drift['drift_share']*100:.1f}%")
                else:
                    st.metric("Drift share", "—")

            with col3:
                if trust.get("trust_available") and trust.get("trust_pass_rate") is not None:
                    st.metric("Trust pass rate", f"{trust['trust_pass_rate']*100:.1f}%")
                else:
                    st.metric("Trust pass rate", "—")

            if drift.get("drift_available") and drift.get("drifted_columns"):
                st.warning(
                    f"Drifted columns: {', '.join(drift['drifted_columns'])}"
                )
            elif drift.get("drift_available"):
                st.success("No drift detected in any shared column.")

            if drift.get("drift_error"):
                st.caption(f"Note: {drift['drift_error']}")
        elif "drift_reference_df" in st.session_state:
            st.info(
                "Reference dataset is loaded, but the Multi-Agent Analysis "
                "hasn't run yet. Go to the Multi-Agent Analysis page and click "
                "'Run Multi-Agent Analysis' to compute drift and trust results."
            )

    # ════════════════════════════════════════════════════════════════
    # SECTION 1 — MISSING VALUE SUMMARY
    # ════════════════════════════════════════════════════════════════
    total_cells   = df.size
    total_missing = df.isnull().sum().sum()
    overall_pct   = round(total_missing / total_cells * 100, 2)
    cols_affected = (df.isnull().sum() > 0).sum()

    # ── TOP METRICS ──
    st.markdown('<div class="section-header">Dataset Overview</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, f"{len(df):,}",       "Total Rows")
    _metric_card(c2, f"{df.shape[1]}",     "Total Columns")
    _metric_card(c3, f"{total_missing:,}", "Missing Cells")
    _metric_card(c4, f"{overall_pct}%",    "Overall Missing %")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── HEALTH BADGE ──
    if total_missing == 0:
        st.markdown(
            '<span class="clean-badge">✅ Dataset is 100% complete — no missing values found</span>',
            unsafe_allow_html=True)
        return

    badge_txt = f"⚠️ {cols_affected} of {df.shape[1]} columns have missing values"
    st.markdown(f'<span class="warn-badge">{badge_txt}</span>',
                unsafe_allow_html=True)

    # ── MISSING TABLE ──
    st.markdown('<div class="section-header">Missing Value Summary Table</div>',
                unsafe_allow_html=True)
    summary = _missing_table(df)

    # ── FIX 1: plain dataframe, no styler ──
    st.dataframe(summary, width='stretch', hide_index=True)

    # ── BAR CHART ──
    st.markdown('<div class="section-header">Missing % Bar Chart</div>',
                unsafe_allow_html=True)
    # ── FIX 2: width='stretch' replaces use_container_width ──
    st.plotly_chart(_bar_chart(summary), width='stretch')

    # ── HEATMAP ──
    cols_with_missing = summary["Column"].tolist()
    st.markdown('<div class="section-header">Missingness Heatmap</div>',
                unsafe_allow_html=True)
    # ── FIX 3: width='stretch' replaces use_container_width ──
    st.plotly_chart(_heatmap(df, cols_with_missing), width='stretch')

    # ── EXPORT ──
    st.markdown('<div class="section-header">Export Report</div>',
                unsafe_allow_html=True)
    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Missing Value Report (CSV)",
        data=csv,
        file_name="missing_value_report.csv",
        mime="text/csv",
    )

    # ════════════════════════════════════════════════════════════════
    # SECTION 2 — DUPLICATE DETECTOR
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Duplicate Row Detector</div>',
                unsafe_allow_html=True)

    # ── Subset selector ──
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Check duplicates across these columns (leave blank = all columns):",
        options=all_cols,
        default=[],
        key="dup_cols"
    )
    subset = selected_cols if selected_cols else None

    # ── Detect duplicates ──
    dup_mask        = df.duplicated(subset=subset, keep=False)
    dup_rows        = df[dup_mask]
    dup_count       = df.duplicated(subset=subset, keep="first").sum()
    dup_pct         = round(dup_count / len(df) * 100, 2)

    # ── Duplicate metrics ──
    d1, d2, d3 = st.columns(3)
    _metric_card(d1, f"{len(df):,}",    "Total Rows")
    _metric_card(d2, f"{dup_count:,}",  "Duplicate Rows")
    _metric_card(d3, f"{dup_pct}%",     "Duplicate %")

    st.markdown("<br>", unsafe_allow_html=True)

    if dup_count == 0:
        st.markdown(
            '<span class="clean-badge">✅ No duplicate rows found</span>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<span class="warn-badge">⚠️ {dup_count} duplicate rows detected</span>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Show duplicate rows ──
        with st.expander(f"👁️ Preview duplicate rows ({len(dup_rows)} rows)"):
            st.dataframe(dup_rows, width='stretch', hide_index=False)

        # ── Remove duplicates button ──
        st.markdown('<div class="section-header">Remove Duplicates</div>',
                    unsafe_allow_html=True)

        if st.button("🗑️ Remove Duplicate Rows", key="remove_dups"):
            before = len(df)
            clean_df = df.drop_duplicates(subset=subset, keep="first")
            after = len(clean_df)
            removed = before - after

            # Save back to session state
            st.session_state["df"] = clean_df

            st.success(
                f"✅ Done — removed {removed} duplicate rows. "
                f"Dataset now has {after:,} rows."
            )

            # ── Before / After comparison ──
            st.markdown('<div class="section-header">Before / After</div>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            _metric_card(b1, f"{before:,}", "Rows Before")
            _metric_card(b2, f"{after:,}",  "Rows After")

            # ── Download cleaned dataset ──
            st.markdown('<div class="section-header">Download Cleaned Dataset</div>',
                        unsafe_allow_html=True)
            clean_csv = clean_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Deduplicated Dataset (CSV)",
                data=clean_csv,
                file_name="deduplicated_dataset.csv",
                mime="text/csv",
                key="dl_dedup"
            )

    # ════════════════════════════════════════════════════════════════
    # SECTION 3 — DATA TYPE FIXER
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🔧 Data Type Fixer</div>',
                unsafe_allow_html=True)

    # ── Current types table ──
    type_df = pd.DataFrame({
        "Column"        : df.columns.tolist(),
        "Current Type"  : df.dtypes.astype(str).tolist(),
        "Sample Value"  : [str(df[c].dropna().iloc[0])
                           if df[c].dropna().shape[0] > 0
                           else "N/A"
                           for c in df.columns],
        "Unique Values" : [df[c].nunique() for c in df.columns],
    })

    st.markdown('<div class="section-header">Current Column Types</div>',
                unsafe_allow_html=True)
    st.dataframe(type_df, width='stretch', hide_index=True)

    # ── Type conversion UI ──
    st.markdown('<div class="section-header">Convert Column Type</div>',
                unsafe_allow_html=True)

    col_to_convert = st.selectbox(
        "Select column to convert:",
        options=df.columns.tolist(),
        key="type_col"
    )

    target_type = st.selectbox(
        "Convert to type:",
        options=[
            "int64",
            "float64",
            "string",
            "datetime",
            "boolean",
            "category",
        ],
        key="target_type"
    )

    if st.button("⚡ Convert Type", key="convert_type"):
        try:
            df_copy = st.session_state["df"].copy()

            if target_type == "int64":
                df_copy[col_to_convert] = pd.to_numeric(
                    df_copy[col_to_convert], errors="coerce"
                ).astype("Int64")

            elif target_type == "float64":
                df_copy[col_to_convert] = pd.to_numeric(
                    df_copy[col_to_convert], errors="coerce"
                ).astype("float64")

            elif target_type == "string":
                df_copy[col_to_convert] = df_copy[col_to_convert].astype(str)

            elif target_type == "datetime":
                df_copy[col_to_convert] = pd.to_datetime(
                    df_copy[col_to_convert], errors="coerce"
                )

            elif target_type == "boolean":
                df_copy[col_to_convert] = df_copy[col_to_convert].astype(bool)

            elif target_type == "category":
                df_copy[col_to_convert] = df_copy[col_to_convert].astype("category")

            # ── Save back ──
            st.session_state["df"] = df_copy

            # ── Confirm ──
            new_type = str(df_copy[col_to_convert].dtype)
            st.success(
                f"✅ Column '{col_to_convert}' converted to {new_type} successfully."
            )

            # ── Show updated types ──
            updated_type_df = pd.DataFrame({
                "Column"       : df_copy.columns.tolist(),
                "Updated Type" : df_copy.dtypes.astype(str).tolist(),
            })
            st.dataframe(updated_type_df, width='stretch', hide_index=True)

        except Exception as e:
            st.error(f"❌ Conversion failed: {e}")

    # ── Download type-fixed dataset ──
    st.markdown('<div class="section-header">Download Type-Fixed Dataset</div>',
                unsafe_allow_html=True)
    current_df   = st.session_state.get("df", df)
    type_fix_csv = current_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Type-Fixed Dataset (CSV)",
        data=type_fix_csv,
        file_name="type_fixed_dataset.csv",
        mime="text/csv",
        key="dl_typefixed"
    )

    # ════════════════════════════════════════════════════════════════
    # SECTION 4 — AUTO CLEANER
    # ════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Auto Cleaner</div>',
                unsafe_allow_html=True)

    st.info(
        "Auto Cleaner applies a smart cleaning pipeline to your dataset. "
        "Review each option and click **Run Auto Clean** when ready."
    )

    # ── Options ──
    st.markdown('<div class="section-header">Cleaning Options</div>',
                unsafe_allow_html=True)

    opt_col_names   = st.checkbox(
        "✅ Fix column names — strip spaces, lowercase, replace spaces with _",
        value=True, key="opt_col"
    )
    opt_strip_str   = st.checkbox(
        "✅ Strip leading/trailing whitespace from text columns",
        value=True, key="opt_strip"
    )
    opt_fill_num    = st.checkbox(
        "✅ Fill missing numeric values with column MEDIAN",
        value=True, key="opt_num"
    )
    opt_fill_cat    = st.checkbox(
        "✅ Fill missing text/category values with 'Unknown'",
        value=True, key="opt_cat"
    )
    opt_drop_empty  = st.checkbox(
        "⚠️ Drop columns where missing % > 90%",
        value=False, key="opt_drop_col"
    )
    opt_drop_duprow = st.checkbox(
        "✅ Remove duplicate rows",
        value=True, key="opt_dup"
    )

    # ── Run button ──
    if st.button("🚀 Run Auto Clean", key="run_auto_clean"):
        df_clean  = st.session_state.get("df").copy()
        log       = []

        # 1 — Fix column names
        if opt_col_names:
            old_cols = df_clean.columns.tolist()
            df_clean.columns = (
                df_clean.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace(r"[^\w]", "_", regex=True)
            )
            new_cols = df_clean.columns.tolist()
            changed  = sum(o != n for o, n in zip(old_cols, new_cols))
            log.append(f"🔤 Column names fixed — {changed} columns renamed")

        # 2 — Strip whitespace
        if opt_strip_str:
            str_cols = df_clean.select_dtypes(include="object").columns
            for c in str_cols:
                df_clean[c] = df_clean[c].str.strip()
            log.append(f"🧼 Whitespace stripped from {len(str_cols)} text columns")

        # 3 — Fill numeric missing with median
        if opt_fill_num:
            num_cols = df_clean.select_dtypes(include="number").columns
            filled   = 0
            for c in num_cols:
                n = df_clean[c].isnull().sum()
                if n > 0:
                    df_clean[c] = df_clean[c].fillna(df_clean[c].median())
                    filled += n
            log.append(f"🔢 Numeric missing filled with median — {filled} cells fixed")

        # 4 — Fill categorical missing with Unknown
        if opt_fill_cat:
            cat_cols = df_clean.select_dtypes(include="object").columns
            filled   = 0
            for c in cat_cols:
                n = df_clean[c].isnull().sum()
                if n > 0:
                    df_clean[c] = df_clean[c].fillna("Unknown")
                    filled += n
            log.append(f"🔡 Text missing filled with 'Unknown' — {filled} cells fixed")

        # 5 — Drop high-missing columns
        if opt_drop_empty:
            before_cols = df_clean.shape[1]
            thresh      = 0.90 * len(df_clean)
            df_clean    = df_clean.dropna(thresh=int(len(df_clean) - thresh), axis=1)
            dropped     = before_cols - df_clean.shape[1]
            log.append(f"🗑️ Dropped {dropped} columns with >90% missing values")

        # 6 — Remove duplicate rows
        if opt_drop_duprow:
            before_rows = len(df_clean)
            df_clean    = df_clean.drop_duplicates(keep="first")
            removed     = before_rows - len(df_clean)
            log.append(f"♻️ Duplicate rows removed — {removed} rows dropped")

        # ── Save clean df ──
        st.session_state["df"]       = df_clean
        st.session_state["df_clean"] = df_clean

        # ── Show log ──
        st.markdown('<div class="section-header">Cleaning Log</div>',
                    unsafe_allow_html=True)
        for entry in log:
            st.markdown(f"- {entry}")

        # ── Before / After ──
        st.markdown('<div class="section-header">Before / After Summary</div>',
                    unsafe_allow_html=True)
        orig = st.session_state.get("df_original", df_clean)
        a1, a2, a3, a4 = st.columns(4)
        _metric_card(a1, f"{len(df_clean):,}",       "Rows After Clean")
        _metric_card(a2, f"{df_clean.shape[1]}",     "Columns After Clean")
        _metric_card(a3, f"{df_clean.isnull().sum().sum()}", "Missing Cells Left")
        _metric_card(a4, f"{df_clean.duplicated().sum()}", "Duplicates Left")

        st.success("✅ Auto Clean complete — dataset updated in session.")

        # ── Download ──
        st.markdown('<div class="section-header">Download Clean Dataset</div>',
                    unsafe_allow_html=True)
        final_csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Final Clean Dataset (CSV)",
            data=final_csv,
            file_name="clean_dataset.csv",
            mime="text/csv",
            key="dl_final_clean"
        )