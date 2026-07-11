import streamlit as st

st.set_page_config(page_title="About | Disease Predictor", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")
st.divider()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    ## Disease Prediction System
    
    This is an end-to-end Machine Learning web application that predicts 
    likely diseases from patient symptoms using a trained **Random Forest** model 
    with **98.5% accuracy** across 41 diseases.

    ### Problem Statement
    Patients in rural and semi-urban India often lack access to preliminary 
    medical screening. This tool helps users understand potential conditions 
    based on their symptoms before consulting a doctor — acting as a 
    first-level screening assistant.

    ### What makes this project unique?
    - **Real dataset** — Trained on 4,920 patient records with 132 symptom features
    - **SHAP explainability** — Every prediction is explained symptom-by-symptom
    - **Urgency detection** — Flags high-risk conditions automatically  
    - **Live web deployment** — Fully accessible via browser, no installation needed
    - **India-relevant diseases** — Covers Malaria, Dengue, Typhoid, Jaundice etc.
    """)

with col2:
    st.markdown("### Quick Stats")
    st.metric("Model Accuracy",  "98.5%")
    st.metric("Diseases Covered","41")
    st.metric("Symptom Features","132")
    st.metric("Training Samples","4,920")
    st.metric("Algorithm",       "Random Forest")
    st.metric("Explainability",  "SHAP")

st.divider()
st.markdown("### Tech Stack")
cols = st.columns(6)
stack = [
    ("🐍 Python",     "3.11"),
    ("📊 Pandas",     "Data wrangling"),
    ("🤖 Scikit-learn","Random Forest"),
    ("⚡ XGBoost",    "Gradient boosting"),
    ("💡 SHAP",       "Explainability"),
    ("🌐 Streamlit",  "Web deployment"),
]
for col, (name, desc) in zip(cols, stack):
    col.metric(name, desc)

st.divider()
st.markdown("""
### Dataset
- **Source:** [Kaggle – Disease Prediction Using ML](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **Author:** kaushil268
- **Rows:** 4,920 training + 42 test records
- **Target:** 41 disease classes (120 samples each — perfectly balanced)

### How to run locally
```bash
git clone https://github.com/yourusername/disease-predictor
cd disease-predictor
pip install -r requirements.txt
python disease_prediction_model.py   # train & save model
streamlit run app.py                 # launch web app
```

### Disclaimer
⚠️ This application is an **educational ML project** and is not intended to 
replace professional medical diagnosis or treatment. Always consult a qualified 
healthcare professional for any medical concerns.
""")