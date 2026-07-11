# ============================================================
#  utils/shap_utils.py — SHAP explainability helpers
# ============================================================

import numpy as np
import pandas as pd
import shap


def get_shap_explanation(model, input_vec: np.ndarray,
                          symptom_cols: list, pred_class: int) -> pd.DataFrame:
    """
    Compute SHAP values for a single prediction.

    Returns a DataFrame with columns: Symptom, SHAP, Present
    sorted by SHAP value descending.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_vec.reshape(1, -1))

    sv = shap_values[pred_class][0]

    df = pd.DataFrame({
        'Symptom': symptom_cols,
        'SHAP'   : sv,
        'Present': input_vec
    })
    return df.sort_values('SHAP', ascending=False)


def top_contributing_symptoms(shap_df: pd.DataFrame,
                               top_n: int = 15,
                               present_only: bool = True) -> pd.DataFrame:
    """
    Return top N symptoms contributing most to the prediction.
    If present_only=True, only include symptoms the patient has selected.
    """
    if present_only:
        df = shap_df[shap_df['Present'] == 1]
    else:
        df = shap_df
    return df.nlargest(top_n, 'SHAP').copy()


def shap_summary_for_all(model, X_sample: np.ndarray,
                          symptom_cols: list) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values across multiple samples.
    Useful for global feature importance summary.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)   # list of arrays [n_classes, n_samples, n_features]

    # Mean absolute SHAP across classes and samples
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    return pd.DataFrame({
        'Symptom'   : symptom_cols,
        'Mean_SHAP' : mean_abs
    }).sort_values('Mean_SHAP', ascending=False)