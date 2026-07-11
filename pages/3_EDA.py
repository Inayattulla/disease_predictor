import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EDA | Disease Predictor", page_icon="🔬", layout="wide")

st.title("🔬 Data Exploration")
st.caption("Understand the dataset before trusting the model")
st.divider()

@st.cache_data
def load_data():
    train = pd.read_csv('data/Training.csv')
    test  = pd.read_csv('data/Testing.csv')
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]
    test  = test.loc[:, ~test.columns.str.contains('^Unnamed')]
    return train, test

try:
    train_df, test_df = load_data()
    symptom_cols = train_df.columns[:-1].tolist()
    data_loaded  = True
except:
    data_loaded  = False
    st.warning("Dataset not found. Place `Training.csv` and `Testing.csv` in the `data/` folder.")

if data_loaded:
    # ── Dataset summary ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training rows", len(train_df))
    c2.metric("Test rows",     len(test_df))
    c3.metric("Symptom cols",  len(symptom_cols))
    c4.metric("Diseases",      train_df['prognosis'].nunique())
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["Disease Distribution", "Symptom Frequency",
                               "Disease-Symptom Heatmap", "Symptom Search"])

    with t1:
        st.markdown("#### Disease distribution in training set")
        st.caption("Each disease has exactly 120 samples — perfectly balanced dataset.")
        counts = train_df['prognosis'].value_counts().reset_index()
        counts.columns = ['Disease', 'Count']
        fig = px.bar(counts, x='Count', y='Disease', orientation='h',
                     color='Count', color_continuous_scale=['#B5D4F4', '#185FA5'])
        fig.update_layout(height=900, coloraxis_showscale=False,
                          yaxis={'categoryorder':'total ascending'},
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("#### Symptom frequency across all patients")
        top_n = st.slider("Show top N symptoms", 10, 60, 30)
        freq  = train_df[symptom_cols].sum().sort_values(ascending=False).head(top_n).reset_index()
        freq.columns = ['Symptom', 'Frequency']
        freq['Symptom'] = freq['Symptom'].str.replace('_', ' ').str.title()
        fig2 = px.bar(freq, x='Frequency', y='Symptom', orientation='h',
                      color='Frequency', color_continuous_scale=['#B5D4F4', '#1D9E75'])
        fig2.update_layout(height=max(400, top_n * 22), coloraxis_showscale=False,
                           yaxis={'categoryorder':'total ascending'},
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        st.markdown("#### Disease vs symptom prevalence heatmap")
        st.caption("Darker = symptom is more common in that disease")
        top_k = st.slider("Top K symptoms", 15, 50, 30)
        top_syms = train_df[symptom_cols].sum().sort_values(ascending=False).head(top_k).index.tolist()
        heat_data = train_df.groupby('prognosis')[top_syms].mean()
        heat_data.columns = [c.replace('_', ' ').title() for c in heat_data.columns]

        fig3 = px.imshow(
            heat_data,
            aspect='auto',
            color_continuous_scale='Blues',
            title=f'Disease vs Top {top_k} Symptoms'
        )
        fig3.update_layout(height=900,
                           xaxis_tickangle=-45,
                           paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        st.markdown("#### Explore which diseases have a specific symptom")
        search = st.selectbox(
            "Pick a symptom:",
            options=sorted([s.replace('_', ' ').title() for s in symptom_cols])
        )
        sym_col = search.lower().replace(' ', '_')
        if sym_col in symptom_cols:
            diseases_with = train_df[train_df[sym_col] == 1]['prognosis'].value_counts().reset_index()
            diseases_with.columns = ['Disease', 'Patients with symptom']
            if len(diseases_with) == 0:
                st.info("No patients have this symptom in the training set.")
            else:
                st.markdown(f"**{len(diseases_with)} disease(s)** have the symptom **'{search}'**")
                fig4 = px.bar(diseases_with, x='Patients with symptom', y='Disease',
                              orientation='h', color='Patients with symptom',
                              color_continuous_scale=['#FAEEDA', '#E07B00'])
                fig4.update_layout(height=max(300, len(diseases_with) * 30),
                                   coloraxis_showscale=False,
                                   yaxis={'categoryorder': 'total ascending'},
                                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig4, use_container_width=True)