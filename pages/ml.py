# pages/ml.py

import streamlit as st
import pandas as pd
import numpy as np
from config.settings import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SURFACE_COLOR,
    TEXT_COLOR, ACCENT_COLOR
)


# ── CSS ──────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown(f"""
    <style>
      .ml-header {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.2rem;
      }}
      .ml-sub {{
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
      .verdict-ok {{
        display: inline-block;
        background: {ACCENT_COLOR}22;
        border: 1px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 700;
      }}
      .verdict-warn {{
        display: inline-block;
        background: #FF6B6B22;
        border: 1px solid #FF6B6B;
        color: #FF6B6B;
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 700;
      }}
      .info-box {{
        background: {SURFACE_COLOR};
        border-left: 3px solid {PRIMARY_COLOR};
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: {TEXT_COLOR}CC;
        margin: 0.5rem 0 1rem 0;
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


# ── Main entry ───────────────────────────────────────────────────────────────
def show():
    _inject_css()

    st.markdown('<div class="ml-header">🤖 Machine Learning Engine</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="ml-sub">Automated model training, evaluation, '
                'and feature analysis.</div>', unsafe_allow_html=True)

    df = st.session_state["df"].copy()

    # Sanitize
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols     = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for ML. "
                 "Use Data Cleaning → Data Type Fixer first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Problem Setup",
        "📊 Train & Evaluate",
        "🔍 Feature Importance",
        "💾 Export Model"
    ])

    with tab1:
        _problem_setup(df, numeric_cols, all_cols)

    with tab2:
        _train_evaluate(df)

    with tab3:
        _feature_importance()

    with tab4:
        _export_model()


# ── Tab 1: Problem Setup ─────────────────────────────────────────────────────
def _problem_setup(df, numeric_cols, all_cols):
    st.divider()
    st.markdown("### ⚙️ Configure Your ML Problem")

    st.markdown("""
    <div class="info-box">
    Select a <b>target</b> (what you want to predict) and 
    <b>features</b> (what the model learns from). 
    Then choose whether this is a regression or classification problem.
    </div>
    """, unsafe_allow_html=True)

    # ── Target column ────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        target_col = st.selectbox(
            "🎯 Target Column (what to predict)",
            options=all_cols,
            key="ml_target"
        )

    with c2:
        # Auto-detect problem type
        n_unique = df[target_col].nunique()
        if n_unique <= 10:
            default_type = "Classification"
        else:
            default_type = "Regression"

        problem_type = st.selectbox(
            "📌 Problem Type",
            options=["Regression", "Classification"],
            index=0 if default_type == "Regression" else 1,
            key="ml_problem_type"
        )

    # ── Feature columns ──────────────────────────────────────────────────────
    feature_options = [c for c in all_cols if c != target_col]
    default_features = [c for c in numeric_cols if c != target_col]

    selected_features = st.multiselect(
        "📋 Feature Columns (inputs to the model)",
        options=feature_options,
        default=default_features[:min(8, len(default_features))],
        key="ml_features"
    )

    st.divider()

    # ── Train/test split ─────────────────────────────────────────────────────
    st.markdown("### ✂️ Train / Test Split")

    s1, s2, s3 = st.columns(3)

    with s1:
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10, max_value=40,
            value=20, step=5,
            key="ml_test_size"
        )

    with s2:
        random_seed = st.number_input(
            "Random Seed",
            min_value=0, max_value=999,
            value=42, step=1,
            key="ml_seed"
        )

    with s3:
        model_choice = st.selectbox(
            "🤖 Algorithm",
            options=_get_model_options(problem_type),
            key="ml_model_choice"
        )

    st.divider()

    # ── Validation ───────────────────────────────────────────────────────────
    errors   = []
    warnings = []

    if not selected_features:
        errors.append("Select at least one feature column.")

    if target_col in selected_features:
        errors.append("Target column cannot also be a feature.")

    if len(df) < 50:
        warnings.append(f"Small dataset ({len(df)} rows). "
                        "Results may not generalise well.")

    if problem_type == "Classification":
        n_cls = df[target_col].nunique()
        if n_cls > 20:
            warnings.append(f"Target has {n_cls} unique values. "
                            "Are you sure this is classification?")
        if n_cls < 2:
            errors.append("Classification target needs at least 2 classes.")

    # Non-numeric features warning
    non_num = [f for f in selected_features
               if f not in df.select_dtypes(include="number").columns]
    if non_num:
        warnings.append(f"Non-numeric features will be label-encoded: "
                        f"{', '.join(non_num)}")

    for e in errors:
        st.error(f"❌ {e}")
    for w in warnings:
        st.warning(f"⚠️ {w}")

    # ── Summary cards ────────────────────────────────────────────────────────
    if not errors:
        train_rows = int(len(df) * (1 - test_size / 100))
        test_rows  = len(df) - train_rows

        m1, m2, m3, m4 = st.columns(4)
        _metric_card(m1, f"{len(selected_features)}", "Features Selected")
        _metric_card(m2, f"{train_rows:,}",           "Train Rows")
        _metric_card(m3, f"{test_rows:,}",            "Test Rows")
        _metric_card(m4,
                     df[target_col].nunique()
                     if problem_type == "Classification"
                     else f"{df[target_col].dtype}",
                     "Classes" if problem_type == "Classification"
                     else "Target Type")

        st.markdown("")

        # ── Confirm & save config ────────────────────────────────────────────
        if st.button("✅ Confirm Setup & Proceed to Training",
                     width='stretch', key="ml_confirm"):
            st.session_state["ml_config"] = {
                "target":       target_col,
                "features":     selected_features,
                "problem_type": problem_type,
                "test_size":    test_size / 100,
                "random_seed":  int(random_seed),
                "model_choice": model_choice,
                "non_numeric":  non_num
            }
            st.success("✅ Configuration saved. "
                       "Go to **📊 Train & Evaluate** tab.")

        if "ml_config" in st.session_state:
            cfg = st.session_state["ml_config"]
            st.markdown(f"""
            <div class="info-box">
            <b>Active Config:</b> {cfg['problem_type']} · 
            Target: <b>{cfg['target']}</b> · 
            Features: {len(cfg['features'])} · 
            Algorithm: <b>{cfg['model_choice']}</b> · 
            Test: {int(cfg['test_size']*100)}%
            </div>
            """, unsafe_allow_html=True)


# ── Stubs for tabs 2-4 (built in next steps) ─────────────────────────────────
def _train_evaluate(df):
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import (LinearRegression, Ridge,
                                      Lasso, LogisticRegression)
    from sklearn.ensemble import (RandomForestRegressor,
                                  RandomForestClassifier)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (
        r2_score, mean_squared_error,
        accuracy_score, confusion_matrix,
        classification_report, roc_curve, auc
    )

    if "ml_config" not in st.session_state:
        st.info("⚙️ Complete **Problem Setup** first and click Confirm.",
                icon="👆")
        return

    cfg          = st.session_state["ml_config"]
    target       = cfg["target"]
    features     = cfg["features"]
    problem_type = cfg["problem_type"]
    test_size    = cfg["test_size"]
    seed         = cfg["random_seed"]
    model_name   = cfg["model_choice"]

    st.divider()
    st.markdown("### 🚀 Train Model")

    if not st.button("▶️ Run Training", width='stretch',
                     key="ml_run_training"):
        st.info("Click **Run Training** to train your model.", icon="👆")
        return

    # ── Preprocess ───────────────────────────────────────────────────────────
    with st.spinner("Preprocessing data..."):
        work = df[features + [target]].copy()
        work = work.dropna()

        # Encode non-numeric features
        encoders = {}
        for col in features:
            try:
                work[col] = pd.to_numeric(work[col], errors="raise")
            except (ValueError, TypeError):
                le = LabelEncoder()
                work[col] = le.fit_transform(work[col].astype(str))
                encoders[col] = le
            work[col] = work[col].astype(float)

       # Encode target
        target_encoder = None
        try:
            work[target] = pd.to_numeric(work[target], errors="raise")
        except (ValueError, TypeError):
            le = LabelEncoder()
            work[target] = le.fit_transform(work[target].astype(str))
            target_encoder = le
        work[target] = work[target].astype(float)

        X = np.array(work[features].values, dtype=np.float64)
        if problem_type == "Regression":
            y = np.array(work[target].values, dtype=np.float64)
        else:
            y = np.array(work[target].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=seed,
            stratify=y if problem_type == "Classification" else None
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    # ── Build model ──────────────────────────────────────────────────────────
    model_map = {
        "Linear Regression":          LinearRegression(),
        "Ridge Regression":           Ridge(),
        "Lasso Regression":           Lasso(),
        "Random Forest Regressor":    RandomForestRegressor(
                                          n_estimators=100,
                                          random_state=seed),
        "Logistic Regression":        LogisticRegression(
                                          max_iter=1000,
                                          random_state=seed),
        "Random Forest Classifier":   RandomForestClassifier(
                                          n_estimators=100,
                                          random_state=seed),
        "K-Nearest Neighbors":        KNeighborsClassifier(),
        "Decision Tree Classifier":   DecisionTreeClassifier(
                                          random_state=seed)
    }

    model = model_map[model_name]

    with st.spinner(f"Training {model_name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Save model artifacts to session state
    st.session_state["ml_model"]         = model
    st.session_state["ml_scaler"]        = scaler
    st.session_state["ml_encoders"]      = encoders
    st.session_state["ml_target_encoder"]= target_encoder
    st.session_state["ml_features_list"] = features
    st.session_state["ml_X_test"]        = X_test
    st.session_state["ml_y_test"]        = y_test
    st.session_state["ml_y_pred"]        = y_pred

    st.success(f"✅ {model_name} trained successfully.")
    st.divider()

    # ── Metrics ──────────────────────────────────────────────────────────────
    if problem_type == "Regression":
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) /
                              np.where(y_test == 0, 1, y_test))) * 100

        st.markdown("### 📊 Regression Metrics")
        m1, m2, m3, m4 = st.columns(4)
        _metric_card(m1, f"{r2:.4f}",    "R² Score")
        _metric_card(m2, f"{rmse:.4f}",  "RMSE")
        _metric_card(m3, f"{mae:.4f}",   "MAE")
        _metric_card(m4, f"{mape:.2f}%", "MAPE")

        st.markdown("")
        st.markdown("### 🔵 Actual vs Predicted")
        fig = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted Values",
            color_discrete_sequence=[PRIMARY_COLOR]
        )
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                      line=dict(color=ACCENT_COLOR, dash="dash", width=2))
        fig.update_layout(
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title_font_color=PRIMARY_COLOR,
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            margin=dict(t=60, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("### 📉 Residuals Plot")
        residuals = y_test - y_pred
        fig2 = px.scatter(
            x=y_pred, y=residuals,
            labels={"x": "Predicted", "y": "Residual"},
            title="Residuals vs Predicted",
            color_discrete_sequence=[ACCENT_COLOR]
        )
        fig2.add_hline(y=0, line_dash="dash",
                       line_color="#FF6B6B", line_width=1.5)
        fig2.update_layout(
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title_font_color=PRIMARY_COLOR,
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            margin=dict(t=60, b=40, l=40, r=40)
        )
        st.plotly_chart(fig2, width='stretch')

    else:
        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        cr  = classification_report(y_test, y_pred, output_dict=True)

        st.markdown("### 📊 Classification Metrics")
        m1, m2, m3, m4 = st.columns(4)
        _metric_card(m1, f"{acc:.4f}",  "Accuracy")
        _metric_card(m2, f"{cr.get('weighted avg', {}).get('precision', 0):.4f}",
                     "Precision")
        _metric_card(m3, f"{cr.get('weighted avg', {}).get('recall', 0):.4f}",
                     "Recall")
        _metric_card(m4, f"{cr.get('weighted avg', {}).get('f1-score', 0):.4f}",
                     "F1 Score")

        st.markdown("")
        st.markdown("### 🟦 Confusion Matrix")
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix",
            aspect="auto"
        )
        fig.update_layout(
            paper_bgcolor=BACKGROUND_COLOR,
            plot_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            title_font_color=PRIMARY_COLOR,
            margin=dict(t=60, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, width='stretch')

        # ROC curve — binary only
        classes = np.unique(y_test)
        if len(classes) == 2 and hasattr(model, "predict_proba"):
            st.markdown("### 📈 ROC Curve")
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"ROC (AUC = {roc_auc:.4f})",
                line=dict(color=PRIMARY_COLOR, width=2.5)
            ))
            fig3.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random Classifier",
                line=dict(color="#FF6B6B", dash="dash", width=1.5)
            ))
            fig3.update_layout(
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR,
                title="ROC Curve",
                title_font_color=PRIMARY_COLOR,
                xaxis=dict(title="False Positive Rate",
                           gridcolor="#333"),
                yaxis=dict(title="True Positive Rate",
                           gridcolor="#333"),
                legend=dict(bgcolor=SURFACE_COLOR),
                margin=dict(t=60, b=40, l=40, r=40)
            )
            st.plotly_chart(fig3, width='stretch')


def _feature_importance():
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.inspection import permutation_importance

    if "ml_config" not in st.session_state:
        st.info("⚙️ Complete **Problem Setup** first.", icon="👆")
        return

    if "ml_model" not in st.session_state:
        st.info("⚙️ Train a model in **📊 Train & Evaluate** first.",
                icon="👆")
        return

    model    = st.session_state["ml_model"]
    X_test   = st.session_state["ml_X_test"]
    y_test   = st.session_state["ml_y_test"]
    features = st.session_state["ml_features_list"]
    cfg      = st.session_state["ml_config"]

    st.divider()
    st.markdown("### 🔍 Feature Importance Analysis")

    # ── Method selection ─────────────────────────────────────────────────────
    has_native = hasattr(model, "feature_importances_")
    method_options = []
    if has_native:
        method_options.append("Native (Tree-Based Gini/Entropy)")
    method_options.append("Permutation Importance")

    method = st.radio(
        "Importance Method",
        options=method_options,
        horizontal=True,
        key="fi_method"
    )

    st.markdown("")

    # ── Compute importance ───────────────────────────────────────────────────
    with st.spinner("Computing feature importance..."):
        if "Native" in method:
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                "Feature":    features,
                "Importance": importances
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            imp_df["Rank"] = range(1, len(imp_df) + 1)
            method_label = "Gini Importance"

        else:
            seed = cfg["random_seed"]
            perm = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=seed,
                n_jobs=-1
            )
            imp_df = pd.DataFrame({
                "Feature":    features,
                "Importance": perm.importances_mean,
                "Std":        perm.importances_std
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            imp_df["Rank"] = range(1, len(imp_df) + 1)
            method_label = "Permutation Importance"

    # ── Summary cards ────────────────────────────────────────────────────────
    top_feature   = imp_df.iloc[0]["Feature"]
    top_score     = imp_df.iloc[0]["Importance"]
    n_positive    = (imp_df["Importance"] > 0).sum()
    n_features    = len(imp_df)

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, top_feature,        "Most Important Feature")
    _metric_card(m2, f"{top_score:.4f}", "Top Importance Score")
    _metric_card(m3, f"{n_positive}",    "Positively Contributing")
    _metric_card(m4, f"{n_features}",    "Total Features")

    st.markdown("")
    st.divider()

    # ── Horizontal bar chart ─────────────────────────────────────────────────
    st.markdown(f"### 📊 {method_label} Chart")

    chart_df = imp_df.sort_values("Importance", ascending=True)

    fig = go.Figure()

    if "Std" in chart_df.columns:
        fig.add_trace(go.Bar(
            x=chart_df["Importance"],
            y=chart_df["Feature"],
            orientation="h",
            error_x=dict(
                type="data",
                array=chart_df["Std"].tolist(),
                visible=True,
                color="rgba(232,232,240,0.5)"
            ),
            marker=dict(
                color=chart_df["Importance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Score")
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Importance: %{x:.4f}<extra></extra>"
            )
        ))
    else:
        fig.add_trace(go.Bar(
            x=chart_df["Importance"],
            y=chart_df["Feature"],
            orientation="h",
            marker=dict(
                color=chart_df["Importance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Score")
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Importance: %{x:.4f}<extra></extra>"
            )
        ))

    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        title=f"Feature Importance — {method_label}",
        title_font_color=PRIMARY_COLOR,
        xaxis=dict(title="Importance Score", gridcolor="#333"),
        yaxis=dict(title="", gridcolor="#333"),
        margin=dict(t=60, b=40, l=160, r=60),
        height=max(350, len(features) * 40)
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # ── Rankings table ───────────────────────────────────────────────────────
    st.markdown("### 📋 Importance Rankings Table")

    display = imp_df[["Rank", "Feature", "Importance"]].copy()
    if "Std" in imp_df.columns:
        display["Std Dev"] = imp_df["Std"].round(4)
    display["Importance"] = display["Importance"].round(6)

    st.dataframe(display, width='stretch', height=320)

    st.divider()

    # ── Feature-target correlation ───────────────────────────────────────────
    st.markdown("### 🔗 Feature-Target Correlation")
    st.markdown(
        "<div style='font-size:0.8rem; color:#E8E8F088;'>"
        "Pearson correlation between each feature and the target column "
        "(numeric targets only).</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    df_orig = st.session_state["df"].copy()
    target  = cfg["target"]

    if df_orig[target].dtype in ["object", "category"]:
        st.info("Target is categorical — correlation table skipped.")
    else:
        corr_data = []
        for feat in features:
            try:
                col_data = df_orig[feat]
                if col_data.dtype == object:
                    col_data = pd.to_numeric(col_data, errors="coerce")
                corr_val = col_data.corr(df_orig[target])
                corr_data.append({
                    "Feature":     feat,
                    "Correlation": round(corr_val, 4),
                    "Strength":    _corr_label(abs(corr_val))
                })
            except Exception:
                pass

        corr_df = pd.DataFrame(corr_data)
        
        # ── CORRECTED INDENTATION STARTS HERE ──
        if not corr_df.empty and "Correlation" in corr_df.columns:
            corr_df = corr_df.sort_values(
                "Correlation", key=abs, ascending=False
            ).reset_index(drop=True)

            st.dataframe(corr_df, width='stretch', height=280)

            # Correlation bar chart
            fig2 = px.bar(
                corr_df,
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Correlation",
                color_continuous_scale="RdBu",
                range_color=[-1, 1],
                title=f"Feature Correlations with '{target}'",
                text="Correlation"
            )
            fig2.update_traces(texttemplate="%{text:.3f}",
                               textposition="outside")
            fig2.update_layout(
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR,
                title_font_color=PRIMARY_COLOR,
                xaxis=dict(gridcolor="#333", range=[-1.2, 1.2]),
                yaxis=dict(gridcolor="#333"),
                coloraxis_showscale=False,
                margin=dict(t=60, b=40, l=160, r=80),
                height=max(350, len(features) * 40)
            )
            st.plotly_chart(fig2, width='stretch')
        else:
            # Fallback if no correlations could be calculated
            st.info("⚠️ Could not calculate Pearson correlations. This usually happens if your target column contains non-numeric categories that haven't been encoded in the raw dataset.")
    
    # ── Export ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💾 Export")

    from io import BytesIO
    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Importance CSV",
        data=csv_bytes,
        file_name="feature_importance.csv",
        mime="text/csv",
        width='stretch'
    )


def _corr_label(val):
    if val >= 0.7:
        return "Strong"
    elif val >= 0.4:
        return "Moderate"
    elif val >= 0.2:
        return "Weak"
    else:
        return "Negligible"

def _export_model():
    import joblib
    import os
    import json
    from io import BytesIO
    import zipfile

    if "ml_config" not in st.session_state:
        st.info("⚙️ Complete **Problem Setup** first.", icon="👆")
        return

    if "ml_model" not in st.session_state:
        st.info("⚙️ Train a model in **📊 Train & Evaluate** first.",
                icon="👆")
        return

    cfg          = st.session_state["ml_config"]
    model        = st.session_state["ml_model"]
    scaler       = st.session_state["ml_scaler"]
    encoders     = st.session_state["ml_encoders"]
    features     = st.session_state["ml_features_list"]
    problem_type = cfg["problem_type"]
    model_name   = cfg["model_choice"]
    target       = cfg["target"]

    st.divider()
    st.markdown("### 💾 Export Trained Model")

    # ── Model summary ────────────────────────────────────────────────────────
    st.markdown("#### 📋 Model Summary")

    m1, m2, m3, m4 = st.columns(4)
    _metric_card(m1, model_name.split()[0],  "Algorithm Family")
    _metric_card(m2, problem_type,           "Problem Type")
    _metric_card(m3, str(len(features)),     "Features Used")
    _metric_card(m4, target,                 "Target Column")

    st.markdown("")

    # ── Model metadata ───────────────────────────────────────────────────────
    metadata = {
        "model_name":   model_name,
        "problem_type": problem_type,
        "target":       target,
        "features":     features,
        "test_size":    cfg["test_size"],
        "random_seed":  cfg["random_seed"],
        "has_scaler":   True,
        "encoded_cols": list(encoders.keys())
    }

    # ── Inference script ─────────────────────────────────────────────────────
    encoded_cols_str = str(list(encoders.keys()))
    features_str     = str(features)

    inference_script = f'''# AnalyticaOS — Auto-Generated Inference Script
# Model  : {model_name}
# Target : {target}
# Type   : {problem_type}

import joblib
import pandas as pd
import numpy as np

# Load artifacts
model   = joblib.load("model.pkl")
scaler  = joblib.load("scaler.pkl")

# If you had encoded columns, load and apply encoders too
# encoders = joblib.load("encoders.pkl")

FEATURES     = {features_str}
ENCODED_COLS = {encoded_cols_str}

def predict(input_df: pd.DataFrame):
    """
    input_df : DataFrame with columns matching FEATURES
    Returns  : array of predictions
    """
    df = input_df[FEATURES].copy()

    # Encode object columns
    for col in ENCODED_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # Scale
    X = scaler.transform(df.values)

    # Predict
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    # Example usage
    sample = pd.DataFrame([dict.fromkeys(FEATURES, 0)])
    print("Sample prediction:", predict(sample))
'''

    st.divider()
    st.markdown("#### 🐍 Auto-Generated Inference Script")
    st.code(inference_script, language="python")

    st.divider()

    # ── Build ZIP in memory ──────────────────────────────────────────────────
    st.markdown("#### 📦 Download Model Package")
    st.markdown(
        "<div style='font-size:0.82rem; color:#E8E8F088;'>"
        "Downloads a ZIP containing: "
        "<b>model.pkl</b>, <b>scaler.pkl</b>, "
        "<b>encoders.pkl</b>, <b>metadata.json</b>, "
        "<b>inference.py</b></div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    if st.button("📦 Build & Download Model ZIP",
                 width='stretch',
                 key="ml_export_zip"):

        with st.spinner("Packaging model..."):
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, "w",
                                 zipfile.ZIP_DEFLATED) as zf:

                # model.pkl
                model_buf = BytesIO()
                joblib.dump(model, model_buf)
                zf.writestr("model.pkl",
                            model_buf.getvalue())

                # scaler.pkl
                scaler_buf = BytesIO()
                joblib.dump(scaler, scaler_buf)
                zf.writestr("scaler.pkl",
                            scaler_buf.getvalue())

                # encoders.pkl
                enc_buf = BytesIO()
                joblib.dump(encoders, enc_buf)
                zf.writestr("encoders.pkl",
                            enc_buf.getvalue())

                # metadata.json
                zf.writestr("metadata.json",
                            json.dumps(metadata, indent=2))

                # inference.py
                zf.writestr("inference.py", inference_script)

        st.download_button(
            label="⬇️ Download analyticaos_model.zip",
            data=zip_buffer.getvalue(),
            file_name="analyticaos_model.zip",
            mime="application/zip",
            width='stretch',
            key="ml_zip_download"
        )
        st.success("✅ Model package ready. Click the button above to save.")

    st.divider()

    # ── Individual downloads ─────────────────────────────────────────────────
    st.markdown("#### 📄 Individual Downloads")

    col1, col2 = st.columns(2)

    # Inference script
    col1.download_button(
        label="⬇️ Download inference.py",
        data=inference_script.encode("utf-8"),
        file_name="inference.py",
        mime="text/plain",
        width='stretch',
        key="ml_dl_script"
    )

    # Metadata JSON
    col2.download_button(
        label="⬇️ Download metadata.json",
        data=json.dumps(metadata, indent=2).encode("utf-8"),
        file_name="metadata.json",
        mime="application/json",
        width='stretch',
        key="ml_dl_meta"
    )

    st.divider()

    # ── Usage instructions ───────────────────────────────────────────────────
    st.markdown("#### 📖 How to Use the Exported Model")
    st.markdown(f"""
    <div class="info-box">
    <b>1. Unzip</b> the downloaded package into your project folder.<br><br>
    <b>2. Install dependencies:</b><br>
    <code>pip install scikit-learn joblib pandas numpy</code><br><br>
    <b>3. Run inference:</b><br>
    <code>python inference.py</code><br><br>
    <b>4. Or import in your own code:</b><br>
    <code>from inference import predict</code><br>
    <code>predictions = predict(your_dataframe)</code><br><br>
    <b>Features expected:</b> {", ".join(features)}<br>
    <b>Target predicted:</b> {target}
    </div>
    """, unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_model_options(problem_type):
    if problem_type == "Regression":
        return [
            "Linear Regression",
            "Random Forest Regressor",
            "Ridge Regression",
            "Lasso Regression"
        ]
    else:
        return [
            "Logistic Regression",
            "Random Forest Classifier",
            "K-Nearest Neighbors",
            "Decision Tree Classifier"
        ]