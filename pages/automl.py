# pages/automl.py
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from config.settings import PRIMARY_COLOR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def show():
    st.title("🧬 AutoML Engine")
    st.caption("Automated model search and ensembling via AutoGluon")

    df = st.session_state["df"]

    for key in ("automl_config", "automl_predictor", "automl_leaderboard", "automl_test_data"):
        if key not in st.session_state:
            st.session_state[key] = None
    if st.session_state.automl_config is None:
        st.session_state.automl_config = {}

    tab_setup, tab_importance, tab_export = st.tabs(
        ["Setup & Run", "Feature Importance", "Export & Predict"]
    )

    with tab_setup:
        st.subheader("Problem Setup")

        target_col = st.selectbox("Target Column", df.columns, key="automl_target")
        feature_options = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect(
            "Feature Columns", feature_options, default=feature_options, key="automl_features"
        )

        target_series = df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
            problem_type = "regression"
        elif target_series.nunique() == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"
        st.info(f"Detected problem type: **{problem_type}**")

        col1, col2 = st.columns(2)
        with col1:
            time_limit = st.slider("Time Budget (seconds)", 30, 600, 120, step=30)
        with col2:
            preset = st.selectbox(
                "Quality Preset", ["medium_quality", "good_quality", "best_quality"], index=0
            )
        holdout_pct = st.slider("Held-out Test Split (%)", 10, 30, 20, step=5)

        if st.button("Run AutoML", type="primary"):
            if not selected_features:
                st.error("Select at least one feature column.")
                st.stop()

            full_data = df[selected_features + [target_col]].dropna(subset=[target_col])
            
            # --- START OF FIX: Determine if we can safely stratify ---
            if problem_type != "regression":
                class_counts = full_data[target_col].value_counts()
                if class_counts.min() >= 2:
                    stratify_col = full_data[target_col]
                else:
                    stratify_col = None
                    st.warning("⚠️ **Stratified split disabled:** At least one class in your target column has only 1 instance. Falling back to a random split.")
            else:
                stratify_col = None
            # --- END OF FIX ---

            train_data, test_data = train_test_split(
                full_data, test_size=holdout_pct / 100, random_state=42, stratify=stratify_col
            )

            run_dir = os.path.join(PROJECT_ROOT, "automl_models", f"ag_{int(time.time())}")

            with st.spinner(f"Training models (budget: {time_limit}s)..."):
                predictor = TabularPredictor(
                    label=target_col, problem_type=problem_type, path=run_dir, verbosity=0
                ).fit(train_data, time_limit=time_limit, presets=preset)

            leaderboard = predictor.leaderboard(test_data, silent=True)

            st.session_state.automl_predictor = predictor
            st.session_state.automl_leaderboard = leaderboard
            st.session_state.automl_test_data = test_data
            st.session_state.automl_config = {
                "target": target_col, "features": selected_features,
                "problem_type": problem_type, "run_dir": run_dir,
            }
            st.success(f"Training complete. Best model: **{predictor.model_best}** "
                       f"(scored on {len(test_data)} held-out rows)")

        if st.session_state.automl_leaderboard is not None:
            st.subheader("Leaderboard (held-out test scores)")
            lb = st.session_state.automl_leaderboard.copy()
            lb_display = lb.copy()
            obj_cols = lb_display.select_dtypes(include="object").columns
            lb_display[obj_cols] = lb_display[obj_cols].astype(str)
            st.dataframe(lb_display, width="stretch")

            fig = go.Figure(go.Bar(
                x=lb["score_test"], y=lb["model"], orientation="h", marker_color=PRIMARY_COLOR,
            ))
            fig.update_layout(
                xaxis_title=lb["eval_metric"].iloc[0], yaxis_title="",
                yaxis=dict(autorange="reversed"), height=400,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")

    with tab_importance:
        st.subheader("Feature Importance")
        if st.session_state.automl_predictor is None:
            st.info("Run AutoML in the **Setup & Run** tab first.")
        else:
            if st.button("Compute Feature Importance"):
                with st.spinner("Permutation importance on held-out test set..."):
                    fi = st.session_state.automl_predictor.feature_importance(
                        st.session_state.automl_test_data
                    )
                st.session_state.automl_feature_importance = fi

            fi = st.session_state.get("automl_feature_importance")
            if fi is not None:
                fi_display = fi.reset_index().rename(columns={"index": "feature"})
                st.dataframe(fi_display, width="stretch")
                fig_fi = go.Figure(go.Bar(
                    x=fi["importance"], y=fi_display["feature"], orientation="h",
                    marker_color=PRIMARY_COLOR,
                ))
                fig_fi.update_layout(
                    yaxis=dict(autorange="reversed"), height=400,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_fi, width="stretch")

    with tab_export:
        st.subheader("Export & Predict")
        predictor = st.session_state.automl_predictor
        config = st.session_state.automl_config

        if predictor is None:
            st.info("Run AutoML in the **Setup & Run** tab first.")
        else:
            st.markdown(f"**Best model:** `{predictor.model_best}`")
            st.markdown(f"**Artifact path:** `{config.get('run_dir')}`")
            st.code(
                f'from autogluon.tabular import TabularPredictor\n'
                f'predictor = TabularPredictor.load(r"{config.get("run_dir")}")',
                language="python",
            )

            lb_csv = st.session_state.automl_leaderboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Leaderboard CSV", lb_csv,
                file_name=f"automl_leaderboard_{int(time.time())}.csv", mime="text/csv",
            )

            st.divider()
            st.markdown("**Score new data**")
            new_file = st.file_uploader("Upload CSV to predict on", type=["csv"], key="automl_predict_upload")

            if new_file is not None:
                new_df = pd.read_csv(new_file)
                missing = [c for c in config["features"] if c not in new_df.columns]
                if missing:
                    st.error(f"Uploaded file is missing required columns: {missing}")
                else:
                    preds = predictor.predict(new_df[config["features"]])
                    result_df = new_df.copy()
                    result_df[f"predicted_{config['target']}"] = preds
                    obj_cols = result_df.select_dtypes(include="object").columns
                    display_df = result_df.copy()
                    display_df[obj_cols] = display_df[obj_cols].astype(str)
                    st.dataframe(display_df, width="stretch")

                    pred_csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Predictions CSV", pred_csv,
                        file_name=f"automl_predictions_{int(time.time())}.csv", mime="text/csv",
                    )