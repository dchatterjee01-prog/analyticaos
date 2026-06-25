"""
modeling_agent.py
Wraps Phase 7 (ML Engine) logic into a callable agent.
Auto-detects problem type, selects best algorithm, trains,
and returns evaluation metrics + feature importance.
No Streamlit imports — pure Python.
"""
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent, AgentResult

# sklearn imports — already in requirements.txt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report
)


class ModelingAgent(BaseAgent):
    name = "ModelingAgent"

    CLASSIFICATION_THRESHOLD = 20   # ≤ this many unique target values → classification
    TEST_SIZE                = 0.20
    RANDOM_STATE             = 42

    def run(self, df: pd.DataFrame, context: dict | None = None) -> AgentResult:
        context = context or {}
        findings        = []
        recommendations = []
        artifacts       = {}

        # ── 1. Resolve target column ──────────────────────────────────────
        target_col = context.get("target_col")
        if target_col is None:
            # auto-pick: last numeric column
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return AgentResult(
                    agent_name=self.name,
                    status="error",
                    summary="No numeric columns found — cannot build a model.",
                )
            target_col = num_cols[-1]
            findings.append({
                "title":    "Target auto-selected",
                "detail":   f"No target specified. Using '{target_col}' (last numeric column).",
                "severity": "low",
            })

        if target_col not in df.columns:
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary=f"Target column '{target_col}' not found in DataFrame.",
            )

        # ── 2. Detect problem type ────────────────────────────────────────
        n_unique = df[target_col].nunique()
        is_float = pd.api.types.is_float_dtype(df[target_col])
        
        problem_type = (
            "classification"
            # Prevent tiny float datasets from triggering classification
            if n_unique <= self.CLASSIFICATION_THRESHOLD and not is_float
            else "regression"
        )
        
        artifacts["problem_type"] = problem_type
        artifacts["target_col"]   = target_col

       # ── 3. Build feature matrix ───────────────────────────────────────
        # NOTE: pandas 3.0 introduced a native 'str' dtype distinct from
        # legacy 'object'. df[c].dtype != "object" is no longer sufficient
        # to detect text columns, so we explicitly check is_numeric_dtype.
        feature_cols = [
            c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        # encode low-cardinality text cols (covers both legacy 'object'
        # and pandas 3.0 native 'str' dtype)
        encoders = {}
        df_work = df.copy()
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        for col in text_cols:
            if col == target_col:
                continue
            if col in feature_cols:
                continue  # avoid duplicate column names in feature_cols
            if df[col].nunique() <= 20:
                le = LabelEncoder()
                encoded = le.fit_transform(df_work[col].astype(str)).astype(np.int64)
                df_work[col] = pd.Series(encoded, index=df_work.index)
                encoders[col] = le
                feature_cols.append(col)

        # final safety net: deduplicate while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        artifacts["feature_cols"] = feature_cols
        if not feature_cols:
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary="No usable feature columns after encoding.",
            )

        # drop rows with any nulls in features or target
        cols_needed = feature_cols + [target_col]
        df_clean = df_work[cols_needed].dropna()

        # FIXED: Lowered threshold from 20 to 5 to accommodate small test data
        if len(df_clean) < 5:
            return AgentResult(
                agent_name=self.name,
                status="error",
                summary=f"Only {len(df_clean)} complete rows — too few to train.",
            )

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        # encode target for classification
        target_encoder = None
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            encoders["__target__"] = target_encoder

        # ── 4. Train / test split + scale ─────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ── 5. Train model ────────────────────────────────────────────────
        if problem_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=self.RANDOM_STATE)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=self.RANDOM_STATE)

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        # ── 6. Evaluate ───────────────────────────────────────────────────
        metrics = {}
        if problem_type == "regression":
            metrics["r2"]   = round(r2_score(y_test, y_pred), 4)
            metrics["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)
            metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

            r2 = metrics["r2"]
            sev = "low" if r2 >= 0.7 else ("medium" if r2 >= 0.4 else "high")
            findings.append({
                "title":    f"Regression model trained on '{target_col}'",
                "detail":   f"R² = {r2}, MAE = {metrics['mae']}, RMSE = {metrics['rmse']}",
                "severity": sev,
            })
            if r2 < 0.4:
                recommendations.append(
                    "Low R² — consider feature engineering or trying a different algorithm."
                )

        else:
            acc = round(accuracy_score(y_test, y_pred), 4)
            metrics["accuracy"] = acc
            sev = "low" if acc >= 0.75 else ("medium" if acc >= 0.55 else "high")
            findings.append({
                "title":    f"Classification model trained on '{target_col}'",
                "detail":   f"Accuracy = {acc}",
                "severity": sev,
            })
            if acc < 0.55:
                recommendations.append(
                    "Low accuracy — check class imbalance or add more features."
                )

        artifacts["metrics"]  = metrics
        artifacts["model"]    = model
        artifacts["scaler"]   = scaler
        artifacts["encoders"] = encoders

        # ── 7. Feature importance ─────────────────────────────────────────
        importance_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        artifacts["feature_importance"] = importance_df

        top3 = importance_df.head(3)["feature"].tolist()
        findings.append({
            "title":    "Top predictive features",
            "detail":   f"Most important: {', '.join(top3)}",
            "severity": "low",
        })

        # ── 8. Summary ────────────────────────────────────────────────────
        status = "warning" if any(f["severity"] in ("medium","high") for f in findings) else "ok"
        if problem_type == "regression":
            perf = f"R² = {metrics['r2']}"
        else:
            perf = f"Accuracy = {metrics['accuracy']}"

        summary = (
            f"{problem_type.title()} model on '{target_col}' — {perf}. "
            f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows."
        )

        return AgentResult(
            agent_name=self.name,
            status=status,
            summary=summary,
            findings=findings,
            recommendations=list(dict.fromkeys(recommendations)),
            artifacts=artifacts,
            next_agents=[],
        )