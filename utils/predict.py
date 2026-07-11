# ============================================================
#  utils/predict.py — Prediction helpers used by app.py
# ============================================================

import numpy as np
import pandas as pd
import joblib
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODEL_PATH, ENCODER_PATH, SYMPTOM_PATH,
    URGENT_DISEASES, LOW_CONFIDENCE_THRESHOLD
)


def load_artifacts():
    """Load model, label encoder and symptom list from disk."""
    model    = joblib.load(MODEL_PATH)
    le       = joblib.load(ENCODER_PATH)
    symptoms = joblib.load(SYMPTOM_PATH)
    return model, le, symptoms


def build_input_vector(selected_symptoms: list, symptom_cols: list) -> np.ndarray:
    """Convert list of symptom strings to a binary feature vector."""
    vec = np.zeros(len(symptom_cols))
    unrecognised = []
    for s in selected_symptoms:
        s_clean = s.strip().lower().replace(' ', '_')
        if s_clean in symptom_cols:
            vec[symptom_cols.index(s_clean)] = 1
        else:
            unrecognised.append(s)
    if unrecognised:
        print(f"[warn] Unrecognised symptoms skipped: {unrecognised}")
    return vec


def predict_disease(selected_symptoms: list, model, le, symptom_cols: list) -> dict:
    """
    Full prediction pipeline.

    Returns
    -------
    dict:
        disease     : str   — predicted disease name
        confidence  : float — prediction confidence in %
        top5        : pd.DataFrame — top 5 disease probabilities
        urgent      : bool  — whether urgent medical attention is flagged
        input_vec   : np.ndarray — binary feature vector used for prediction
    """
    vec      = build_input_vector(selected_symptoms, symptom_cols)
    pred_idx = model.predict([vec])[0]
    proba    = model.predict_proba([vec])[0]
    disease  = le.inverse_transform([pred_idx])[0]
    conf     = proba.max()

    top5_idx = proba.argsort()[-5:][::-1]
    top5_df  = pd.DataFrame({
        'Disease'    : [le.classes_[i] for i in top5_idx],
        'Probability': [round(proba[i] * 100, 1) for i in top5_idx]
    })

    urgent = (disease in URGENT_DISEASES) or (conf < LOW_CONFIDENCE_THRESHOLD)

    return {
        'disease'   : disease,
        'confidence': round(conf * 100, 1),
        'top5'      : top5_df,
        'urgent'    : urgent,
        'input_vec' : vec
    }


def fmt_symptom(s: str) -> str:
    """Convert snake_case symptom to Title Case display name."""
    return s.replace('_', ' ').title()


def symptom_display_map(symptom_cols: list) -> dict:
    """Return {display_name: column_name} mapping."""
    return {fmt_symptom(s): s for s in symptom_cols}