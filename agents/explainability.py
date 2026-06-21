"""
explainability.py — Stage A / Step Group 2
Pure-logic SHAP computation, reusing already-trained models from
session_state (Phase 7 / ml.py, or agents/modeling_agent.py artifacts).
No Streamlit imports — testable independently, same pattern as
report_builder.py and the agents/ package.

VERIFIED against shap==0.52.0 (see Stage A Step Group 2, Step 1):
  - Regression:      shap_values.values.shape == (n_samples, n_features)
  - Classification:  shap_values.values.shape == (n_samples, n_features, n_classes)
                      must slice [:, :, class_idx] before plotting or
                      aggregating — the last axis is NOT noise, it's classes.
"""
import numpy as np
import pandas as pd
import shap


class ExplainabilityError(Exception):
    """Raised when SHAP computation cannot proceed (e.g. unsupported model)."""
    pass


def compute_shap_values(model, X: pd.DataFrame, problem_type: str):
    """
    Computes SHAP values for an already-trained tree-based model.

    Args:
        model: fitted sklearn estimator. Must support shap.TreeExplainer —
               in pages/ml.py's model_map this means Random Forest Regressor,
               Random Forest Classifier, or Decision Tree Classifier only.
               (Linear/Ridge/Lasso/Logistic Regression and KNN raise
               shap's InvalidModelError — caught below and re-raised as
               ExplainabilityError.)
        X: feature dataframe, already scaled/encoded exactly as the model
           expects (caller applies session_state["ml_scaler"] / ["ml_encoders"]
           first — this function does not do that itself).
        problem_type: accepts "regression"/"classification" (lowercase,
           agents/modeling_agent.py convention) OR "Regression"/"Classification"
           (Title Case, pages/ml.py's ml_config["problem_type"] convention) —
           normalized internally so callers from either module work unmodified.

    Returns:
        dict with keys:
          shap_explanation: the raw shap.Explanation object (for plotting)
          predicted_classes: np.ndarray of model.predict(X) if classification,
                              else None
          n_classes: int or None (regression)
          is_classification: bool
    """
    if X.empty:
        raise ExplainabilityError("Cannot compute SHAP values on an empty feature set.")

    try:
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X)
    except Exception as e:
        raise ExplainabilityError(
            f"SHAP TreeExplainer failed — this usually means the trained "
            f"model isn't tree-based (only Random Forest and Decision Tree "
            f"models support SHAP explanations here). Original error: {e}"
        ) from e

    is_classification = str(problem_type).strip().lower() == "classification"
    predicted_classes = None
    n_classes = None

    if is_classification:
        if shap_explanation.values.ndim != 3:
            raise ExplainabilityError(
                f"Expected 3D SHAP values for classification, got shape "
                f"{shap_explanation.values.shape}. Model may not be a "
                f"standard sklearn classifier."
            )
        n_classes = shap_explanation.values.shape[2]
        predicted_classes = np.asarray(model.predict(X))
    else:
        if shap_explanation.values.ndim != 2:
            raise ExplainabilityError(
                f"Expected 2D SHAP values for regression, got shape "
                f"{shap_explanation.values.shape}. Model may not be a "
                f"standard sklearn regressor."
            )

    return {
        "shap_explanation": shap_explanation,
        "predicted_classes": predicted_classes,
        "n_classes": n_classes,
        "is_classification": is_classification,
    }


def get_population_view(shap_result: dict, class_idx: int = None):
    shap_explanation = shap_result["shap_explanation"]

    if shap_result["is_classification"]:
        if class_idx is None:
            raise ExplainabilityError(
                "class_idx is required for classification population view."
            )
        if not (0 <= class_idx < shap_result["n_classes"]):
            raise ExplainabilityError(
                f"class_idx {class_idx} out of range for {shap_result['n_classes']} classes."
            )
        return shap_explanation[:, :, class_idx]

    return shap_explanation


def get_row_view(shap_result: dict, row_idx: int):
    shap_explanation = shap_result["shap_explanation"]

    if shap_result["is_classification"]:
        predicted_class = int(shap_result["predicted_classes"][row_idx])
        return shap_explanation[row_idx, :, predicted_class], predicted_class

    return shap_explanation[row_idx], None


def get_mean_abs_importance(shap_result: dict, feature_names: list, class_idx: int = None) -> pd.DataFrame:
    population_view = get_population_view(shap_result, class_idx=class_idx)
    mean_abs = np.abs(population_view.values).mean(axis=0)

    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )