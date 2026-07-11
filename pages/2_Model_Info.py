import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

st.set_page_config(page_title="Model Info | Disease Predictor", page_icon="📊", layout="wide")

st.title("📊 Model Performance")
st.caption("Accuracy metrics, model comparison and feature insights")
st.divider()

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_all():
    model    = joblib.load('models/disease_model.pkl')
    le       = joblib.load('models/label_encoder.pkl')
    symptoms = joblib.load('models/symptom_list.pkl')
    return model, le, symptoms

try:
    model, le, symptom_cols = load_all()
    loaded = True
except:
    loaded = False

# ── Model comparison ──────────────────────────────────────────
st.markdown("### Algorithm Comparison")

comparison_data = {
    'Algorithm'   : ['Naive Bayes (baseline)', 'XGBoost', 'Random Forest'],
    'Test Acc %'  : [85.7, 97.6, 98.5],
    'CV Acc %'    : [85.2, 97.1, 98.3],
    'Train Time'  : ['< 1s', '~10s', '~15s'],
    'Interpretable': ['Yes', 'With SHAP', 'With SHAP'],
    'Best For'    : ['Baseline', 'Production alt.', 'Final model ✓']
}
df_comp = pd.DataFrame(comparison_data)
st.dataframe(df_comp, use_container_width=True, hide_index=True)

fig = px.bar(
    df_comp, x='Algorithm', y='Test Acc %',
    color='Test Acc %',
    color_continuous_scale=['#B5D4F4', '#185FA5'],
    text='Test Acc %',
    title='Test accuracy by algorithm'
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    height=360, coloraxis_showscale=False,
    yaxis_range=[80, 102],
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Why Random Forest ─────────────────────────────────────────
st.markdown("### Why Random Forest?")
c1, c2, c3 = st.columns(3)
c1.info("**Handles binary features well**\nAll 132 features are 0/1 — RF splits on these natively without scaling.")
c2.info("**Built-in feature importance**\nDirectly tells us which symptoms matter most — no extra tools needed.")
c3.info("**Robust to noise**\nEnsemble of 200 trees avoids overfitting even with many symptom columns.")

st.divider()

# ── Feature importance ────────────────────────────────────────
if loaded:
    st.markdown("### Top 20 Most Important Symptoms")
    st.caption("Based on Random Forest's built-in Gini impurity feature importance")

    importances = pd.Series(model.feature_importances_, index=symptom_cols)
    top20       = importances.sort_values(ascending=False).head(20)

    fig2 = px.bar(
        x=top20.values * 100,
        y=[s.replace('_', ' ').title() for s in top20.index],
        orientation='h',
        labels={'x': 'Importance (%)', 'y': 'Symptom'},
        color=top20.values,
        color_continuous_scale=['#B5D4F4', '#185FA5'],
        title='Feature importance — top 20 symptoms'
    )
    fig2.update_layout(
        height=540, coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Diseases covered ──────────────────────────────────────
    st.markdown("### All 41 Diseases Covered")
    diseases = sorted(le.classes_)
    cols = st.columns(4)
    for i, disease in enumerate(diseases):
        cols[i % 4].markdown(f"- {disease}")
else:
    st.warning("Model not found. Run `disease_prediction_model.py` first to train the model.")

st.divider()

# ── Training details ──────────────────────────────────────────
st.markdown("### Training Details")
details = {
    'Parameter'   : ['Dataset', 'Training samples', 'Test samples', 'Features',
                      'Classes', 'Algorithm', 'n_estimators', 'max_features',
                      'Validation', 'Final accuracy'],
    'Value'       : ['Kaggle Disease Symptom Prediction', '3,936', '984', '132 binary symptoms',
                      '41 diseases', 'Random Forest Classifier', '200', 'sqrt(132) ≈ 11',
                      '5-Fold Stratified CV', '98.5%']
}
st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)