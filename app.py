import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os, sys
sys.path.append(os.path.dirname(__file__))

from config import URGENT_DISEASES, LOW_CONFIDENCE_THRESHOLD, APP_TITLE, APP_ICON
from utils.predict import load_artifacts, predict_disease, fmt_symptom, symptom_display_map

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON,
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-title { font-size:2.2rem; font-weight:700; color:#0d1b2a; letter-spacing:-0.5px; }
.subtitle   { font-size:1rem; color:#6b7c93; margin-bottom:1.5rem; }
.result-card{ background:linear-gradient(135deg,#f0f7ff,#e8f4fd); border-left:5px solid #2979d0;
              border-radius:12px; padding:1.4rem 1.8rem; margin:1rem 0; }
.urgent-card{ background:linear-gradient(135deg,#fff5f5,#fdeaea); border-left:5px solid #d63031;
              border-radius:12px; padding:1.4rem 1.8rem; margin:1rem 0; }
.safe-card  { background:linear-gradient(135deg,#f0fff4,#e6f7ed); border-left:5px solid #00b894;
              border-radius:12px; padding:1.4rem 1.8rem; margin:1rem 0; }
.stat-box   { background:white; border:1px solid #e9ecef; border-radius:12px;
              padding:1.2rem 1rem; text-align:center; }
.stat-num   { font-size:2rem; font-weight:700; color:#2979d0; line-height:1; }
.stat-label { font-size:0.82rem; color:#888; margin-top:4px; }
.how-step   { background:#f8f9fa; border-radius:10px; padding:1rem 1.2rem; }
.how-step h4{ font-size:.95rem; font-weight:600; color:#0d1b2a; margin-bottom:6px; }
.how-step p { font-size:.85rem; color:#6b7c93; margin:0; line-height:1.5; }
.disclaimer { font-size:.75rem; color:#adb5bd; border-top:1px solid #f0f0f0;
              padding-top:1rem; margin-top:2.5rem; line-height:1.6; }
[data-testid="stSidebar"] { background:#0d1b2a; }
[data-testid="stSidebar"] * { color:#e8edf2 !important; }
[data-testid="stSidebar"] hr { border-color:#1e3048 !important; }
[data-testid="stSidebar"] .stButton button {
    background:#2979d0 !important; color:white !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_artifacts():
    return load_artifacts()

try:
    model, le, symptom_cols = get_artifacts()
    MODEL_OK = True
except FileNotFoundError:
    MODEL_OK = False

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Disease Predictor")
    st.markdown("<small style='color:#6b8ba4'>Random Forest + SHAP Explainability</small>",
                unsafe_allow_html=True)
    st.markdown("---")
    if not MODEL_OK:
        st.error("Model files not found.\nRun `disease_prediction_model.py` first.")
        st.stop()

    disp_map   = symptom_display_map(symptom_cols)
    disp_names = sorted(disp_map.keys())

    st.markdown("#### Your Symptoms")
    selected_display = st.multiselect(
        "Select all that apply:",
        options=disp_names,
        placeholder="Type to search symptoms..."
    )
    selected = [disp_map[d] for d in selected_display]

    if selected:
        st.markdown(f"<small style='color:#6b8ba4'>{len(selected)} symptom(s) selected</small>",
                    unsafe_allow_html=True)
    st.markdown("---")
    predict_btn = st.button("🔍  Predict Disease", use_container_width=True)
    if st.button("✕  Clear", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.markdown("<small style='color:#6b8ba4'>⚠️ Educational project only — not medical advice.</small>",
                unsafe_allow_html=True)

# ── Main ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🩺 Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered symptom analysis · 41 diseases · 98.5% accuracy · SHAP explanations</div>',
            unsafe_allow_html=True)

# ── Landing ──────────────────────────────────────────────────
if not predict_btn and not selected:
    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl in zip([c1,c2,c3,c4],
                              ["41","132","98.5%","4,920"],
                              ["Diseases","Symptoms","Accuracy","Patient records"]):
        col.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div>'
                     f'<div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How It Works")
    h1,h2,h3 = st.columns(3)
    for col, step, title, desc in zip(
        [h1,h2,h3],
        ["①","②","③"],
        ["Select Symptoms","Click Predict","See Explanation"],
        ["Search and pick every symptom you are experiencing in the sidebar.",
         "Random Forest (200 trees, 98.5% accuracy) analyses your symptom pattern.",
         "Get predicted condition, confidence score, top-5 and SHAP explanation."]
    ):
        col.markdown(f'<div class="how-step"><h4>{step} {title}</h4><p>{desc}</p></div>',
                     unsafe_allow_html=True)
    st.divider()
    st.markdown("### Diseases Covered")
    cols = st.columns(5)
    for i, d in enumerate(sorted(le.classes_)):
        cols[i%5].markdown(f"<small>• {d}</small>", unsafe_allow_html=True)

elif predict_btn and not selected:
    st.warning("⚠️ Please select at least one symptom from the sidebar.")

else:
    result     = predict_disease(selected, model, le, symptom_cols)
    disease    = result['disease']
    confidence = result['confidence']
    top5_df    = result['top5']
    urgent     = result['urgent']
    input_vec  = result['input_vec']

    st.markdown("### Prediction Results")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Predicted Disease", disease)
    m2.metric("Confidence",        f"{confidence}%")
    m3.metric("Symptoms Selected", len(selected))
    m4.metric("Urgent Flag",       "⚠️ Yes" if urgent else "✅ No")

    # ── Disease information database ──────────────────────────
    DISEASE_INFO = {
        'Fungal infection':            {'desc':'A fungal skin or nail infection caused by dermatophytes or yeasts.',          'dos':['Keep affected area clean and dry','Use antifungal cream or powder','Wear loose, breathable clothing'],          'donts':['Do not share towels or footwear','Avoid scratching the affected area','Do not wear tight synthetic clothes'],           'specialist':'Dermatologist',        'icon':'🍄'},
        'Allergy':                     {'desc':'An immune response to allergens like dust, pollen, food, or pet dander.',     'dos':['Identify and avoid triggers','Take antihistamines as directed','Use air purifiers indoors'],                   'donts':['Do not expose yourself to known allergens','Avoid self-medicating without diagnosis','Do not ignore severe reactions'],          'specialist':'Allergist',             'icon':'🤧'},
        'GERD':                        {'desc':'Gastroesophageal reflux disease — stomach acid flows back into the oesophagus.','dos':['Eat smaller, frequent meals','Avoid lying down right after eating','Elevate head while sleeping'],           'donts':["Don't eat spicy or fatty foods","Avoid caffeine and alcohol","Don't wear tight clothing around the waist"], 'specialist':'Gastroenterologist',    'icon':'🔥'},
        'Chronic cholestasis':         {'desc':'Reduced or stopped bile flow from the liver, causing itching and jaundice.',  'dos':['Follow a low-fat diet','Take prescribed bile acid medications','Monitor liver function regularly'],           'donts':['Avoid alcohol completely','Do not take herbal remedies without doctor approval','Avoid fatty foods'],                         'specialist':'Hepatologist',          'icon':'🫀'},
        'Drug Reaction':               {'desc':'An adverse reaction of the body to a medication.',                            'dos':['Stop the suspected drug immediately','Consult your doctor right away','Document the drug name and reaction'], 'donts':['Do not take the same drug again','Avoid self-medication','Do not ignore mild reactions — they can worsen'],    'specialist':'Physician / Allergist', 'icon':'💊'},
        'Peptic ulcer disease':        {'desc':'Painful sores (ulcers) in the lining of the stomach or duodenum.',           'dos':['Take prescribed antacids or PPIs','Eat regular small meals','Reduce stress levels'],                        'donts':['Avoid NSAIDs like ibuprofen','No alcohol or smoking','Avoid spicy and acidic foods'],                             'specialist':'Gastroenterologist',    'icon':'🫃'},
        'AIDS':                        {'desc':'Advanced stage of HIV infection affecting the immune system severely.',        'dos':['Start antiretroviral therapy (ART) immediately','Eat a nutritious diet','Attend regular medical check-ups'],   'donts':['Do not skip ART medications','Avoid unprotected contact','Do not donate blood or organs'],                        'specialist':'Infectious Disease Specialist','icon':'🔴'},
        'Diabetes':                    {'desc':'A metabolic disorder with elevated blood sugar due to insulin issues.',       'dos':['Monitor blood sugar daily','Follow a low-GI diet','Exercise 30 min/day'],                                   'donts':['Avoid sugary drinks and processed foods','Do not skip meals','Avoid sedentary lifestyle'],                         'specialist':'Endocrinologist',       'icon':'🩸'},
        'Gastroenteritis':             {'desc':'Inflammation of the stomach and intestines, usually due to infection.',       'dos':['Drink plenty of oral rehydration solution (ORS)','Rest adequately','Eat light foods like khichdi/rice'],    'donts':["Don't eat dairy or oily food","Avoid solid food until vomiting stops","Don't share utensils"],                    'specialist':'General Physician',     'icon':'🦠'},
        'Bronchial Asthma':            {'desc':'Chronic inflammation of airways causing wheezing and breathing difficulty.',  'dos':['Use prescribed inhalers regularly','Identify and avoid triggers','Practice breathing exercises'],            'donts':['Avoid smoke and dust exposure','Do not exercise without inhaler','Avoid cold air without a mask'],                 'specialist':'Pulmonologist',         'icon':'🫁'},
        'Hypertension':                {'desc':'Persistently high blood pressure (≥140/90 mmHg) stressing the heart.',       'dos':['Monitor BP regularly','Reduce salt intake','Exercise and maintain healthy weight'],                        'donts':['Avoid smoking and alcohol','No excessive caffeine','Do not stop medication without doctor advice'],                'specialist':'Cardiologist',          'icon':'❤️'},
        'Migraine':                    {'desc':'Recurring intense headaches, often with nausea and light sensitivity.',       'dos':['Rest in a dark, quiet room','Stay hydrated','Track migraine triggers in a diary'],                         'donts':['Avoid screen exposure during attacks','Do not skip sleep','Avoid known trigger foods (cheese, alcohol)'],          'specialist':'Neurologist',           'icon':'🧠'},
        'Cervical spondylosis':        {'desc':'Age-related wear of spinal discs in the neck causing stiffness and pain.',    'dos':['Do prescribed physiotherapy exercises','Use ergonomic furniture','Apply heat/cold packs'],                    'donts':['Avoid sudden neck movements','Do not sit in one position for hours','Avoid heavy lifting'],                       'specialist':'Orthopaedist',          'icon':'🦴'},
        'Paralysis (brain hemorrhage)':{'desc':'Loss of muscle function due to brain bleeding disrupting nerve signals.',     'dos':['Seek emergency medical care immediately','Begin rehabilitation as early as possible','Follow physiotherapy'],  'donts':['Do not delay treatment — every minute matters','Avoid strenuous activity','Do not miss follow-up appointments'],   'specialist':'Neurologist (URGENT)',   'icon':'🚨'},
        'Jaundice':                    {'desc':'Yellowing of skin and eyes due to high bilirubin, often from liver issues.',  'dos':['Rest completely','Drink plenty of fluids and coconut water','Follow a liver-friendly diet'],                  'donts':['Avoid alcohol and fatty foods','Do not exert yourself','Avoid self-medication'],                                   'specialist':'Hepatologist',          'icon':'🟡'},
        'Malaria':                     {'desc':'Mosquito-borne parasitic infection causing high fever, chills, and sweating.','dos':['Start antimalarial drugs immediately','Use mosquito nets and repellents','Stay hydrated'],                  'donts':['Do not delay treatment — can be fatal','Avoid being outdoors at dusk','Do not skip medication doses'],             'specialist':'Infectious Disease Specialist','icon':'🦟'},
        'Chicken pox':                 {'desc':'Highly contagious viral infection causing itchy blisters over the body.',     'dos':['Stay isolated at home','Apply calamine lotion on blisters','Stay hydrated and rest'],                       'donts':['Do not scratch blisters — risk of scarring','Avoid contact with pregnant women and newborns','No aspirin'],        'specialist':'General Physician',     'icon':'🐔'},
        'Dengue':                      {'desc':'Mosquito-borne viral fever causing high temperature and severe body aches.',  'dos':['Monitor platelet count daily','Drink plenty of fluids (ORS, coconut water)','Rest completely'],              'donts':['Avoid aspirin or ibuprofen — can worsen bleeding','Do not ignore low platelet count','Avoid mosquito bites'],      'specialist':'General Physician (Urgent)','icon':'🦟'},
        'Typhoid':                     {'desc':'Bacterial infection from Salmonella typhi causing prolonged high fever.',     'dos':['Complete the full antibiotic course','Drink boiled/purified water only','Eat soft, easily digestible food'],  'donts':['Do not eat outside food','Avoid raw vegetables/salads','Never skip antibiotic doses'],                             'specialist':'Infectious Disease Specialist','icon':'🌡️'},
        'Hepatitis A':                 {'desc':'Viral liver infection spread through contaminated food or water.',            'dos':['Rest and maintain good hydration','Eat small nutritious meals','Practice strict hand hygiene'],              'donts':['Avoid alcohol completely','Do not eat oily/spicy food','Avoid sharing utensils'],                                  'specialist':'Hepatologist',          'icon':'🫀'},
        'Hepatitis B':                 {'desc':'Viral infection attacking the liver, spread through blood or bodily fluids.', 'dos':['Take prescribed antiviral medications','Get vaccinated (if not infected)','Regular liver monitoring'],       'donts':['Avoid alcohol','Do not share needles or razors','Avoid unprotected contact'],                                      'specialist':'Hepatologist',          'icon':'🫀'},
        'Hepatitis C':                 {'desc':'Blood-borne viral infection causing chronic liver disease.',                  'dos':['Take direct-acting antiviral (DAA) therapy','Monitor liver function','Eat a liver-friendly diet'],           'donts':['Avoid alcohol','Do not share needles','Avoid NSAIDs like ibuprofen'],                                              'specialist':'Hepatologist',          'icon':'🫀'},
        'Hepatitis D':                 {'desc':'Viral liver infection that only occurs in those with Hepatitis B.',           'dos':['Treat underlying Hepatitis B','Regular liver function tests','Maintain healthy diet'],                       'donts':['Avoid alcohol and fatty food','Do not share needles','Avoid hepatotoxic drugs'],                                   'specialist':'Hepatologist',          'icon':'🫀'},
        'Hepatitis E':                 {'desc':'Waterborne viral liver infection, common in areas with poor sanitation.',     'dos':['Drink only purified/boiled water','Rest and take supportive care','Eat light nutritious meals'],             'donts':['Avoid alcohol','Avoid raw food from unknown sources','Do not take paracetamol in excess'],                         'specialist':'Hepatologist',          'icon':'🫀'},
        'Alcoholic hepatitis':         {'desc':'Liver inflammation caused by excessive alcohol consumption.',                 'dos':['Stop alcohol consumption immediately','Nutritional therapy and supplementation','Medical monitoring'],       'donts':['Never drink alcohol','Do not take paracetamol','Avoid fatty foods'],                                               'specialist':'Hepatologist (Urgent)', 'icon':'🍺'},
        'Tuberculosis':                {'desc':'Bacterial lung infection caused by Mycobacterium tuberculosis.',              'dos':['Complete the full 6-month DOTS treatment','Wear a mask in public','Eat protein-rich foods'],                 'donts':['Never miss medication doses — drug resistance risk','Avoid crowded closed spaces','Do not share utensils'],         'specialist':'Pulmonologist (Urgent)','icon':'🫁'},
        'Common Cold':                 {'desc':'Viral upper respiratory infection causing runny nose, sore throat, sneezing.','dos':['Rest well and stay warm','Drink hot fluids like ginger tea','Use steam inhalation'],                      'donts':["Don't share utensils",'Avoid cold drinks','Do not take antibiotics — it is viral'],                               'specialist':'General Physician',     'icon':'🤒'},
        'Pneumonia':                   {'desc':'Lung infection causing the air sacs to fill with fluid, causing breathing difficulty.','dos':['Take prescribed antibiotics fully','Rest and stay hydrated','Monitor oxygen levels'],          'donts':['Do not delay treatment','Avoid smoking','Avoid cold environments'],                                                'specialist':'Pulmonologist',         'icon':'🫁'},
        'Dimorphic hemmorhoids(piles)':{'desc':'Swollen veins in the rectum or anus causing pain, bleeding, and discomfort.','dos':['Eat high-fibre diet (fruits, vegetables, whole grains)','Drink 8+ glasses of water daily','Sitz baths for relief'], 'donts':['Avoid straining during bowel movements','No spicy or junk food','Avoid prolonged sitting'],              'specialist':'Proctologist',          'icon':'🔺'},
        'Heart attack':                {'desc':'Blockage of blood supply to the heart — a life-threatening emergency.',      'dos':['Call emergency services (108) IMMEDIATELY','Chew aspirin if available','Rest and stay calm'],               'donts':['Do not delay — every second counts','Avoid exertion','Do not eat or drink anything'],                             'specialist':'Cardiologist (EMERGENCY)','icon':'🚨'},
        'Varicose veins':              {'desc':'Enlarged, twisted veins — usually in the legs — causing aching and swelling.','dos':['Elevate legs while resting','Wear compression stockings','Walk regularly'],                             'donts':['Avoid standing or sitting for long periods','Do not cross your legs','Avoid high heels'],                          'specialist':'Vascular Surgeon',      'icon':'🦵'},
        'Hypothyroidism':              {'desc':'Underactive thyroid gland failing to produce enough thyroid hormones.',       'dos':['Take levothyroxine as prescribed','Get TSH levels checked regularly','Eat iodine-rich foods'],              'donts':['Do not skip thyroid medication','Avoid soy products near medication time','Avoid excessive raw cruciferous veg'], 'specialist':'Endocrinologist',       'icon':'🦋'},
        'Hyperthyroidism':             {'desc':'Overactive thyroid producing excess hormones causing rapid heartbeat and weight loss.','dos':['Take antithyroid medications as prescribed','Get regular thyroid function tests','Manage stress'], 'donts':['Avoid iodine-rich foods (seaweed, excess salt)','Do not skip medications','Avoid caffeine'],                    'specialist':'Endocrinologist',       'icon':'🦋'},
        'Hypoglycemia':                {'desc':'Low blood sugar (< 70 mg/dL) causing dizziness, sweating, and confusion.',   'dos':['Consume 15g fast-acting sugar immediately (glucose tabs, juice)','Recheck sugar after 15 minutes','Eat regular meals'], 'donts':['Do not skip meals','Avoid alcohol on empty stomach','Do not delay treating low sugar symptoms'],           'specialist':'Endocrinologist',       'icon':'🩸'},
        'Osteoarthritis':              {'desc':'Degenerative joint disease causing cartilage breakdown, pain, and stiffness.','dos':['Do low-impact exercise (swimming, walking)','Maintain healthy weight','Use hot/cold therapy for pain'],  'donts':['Avoid high-impact activities','Do not stay sedentary','Avoid processed and sugary foods'],                         'specialist':'Rheumatologist',        'icon':'🦴'},
        'Arthritis':                   {'desc':'Inflammation of one or more joints causing pain, swelling, and stiffness.',   'dos':['Do gentle range-of-motion exercises','Take prescribed anti-inflammatory medication','Rest inflamed joints'],  'donts':['Avoid overexertion of joints','Do not ignore worsening pain','Avoid cold damp environments'],                     'specialist':'Rheumatologist',        'icon':'🦴'},
        '(vertigo) Paroymsal Positional Vertigo':{'desc':'Sudden spinning dizziness triggered by head position changes.',   'dos':['Perform Epley manoeuvre (with guidance)','Move slowly when changing positions','Rest if dizzy'],              'donts':['Avoid sudden head movements','Do not drive during episodes','Avoid heights'],                                      'specialist':'ENT Specialist',        'icon':'🌀'},
        'Acne':                        {'desc':'Skin condition with pimples, blackheads, and cysts due to clogged pores.',    'dos':['Wash face twice daily with mild cleanser','Use non-comedogenic moisturiser','Stay hydrated'],               'donts':['Do not pop or squeeze pimples','Avoid oily and greasy skincare products','Avoid touching face frequently'],        'specialist':'Dermatologist',         'icon':'😣'},
        'Urinary tract infection':     {'desc':'Bacterial infection of the bladder or urethra causing burning urination.',    'dos':['Drink 2–3 litres of water daily','Complete the full antibiotic course','Urinate after intercourse'],         'donts':['Avoid holding urine for long','Do not use scented hygiene products','Avoid caffeine and alcohol'],                 'specialist':'Urologist',             'icon':'💧'},
        'Psoriasis':                   {'desc':'Chronic autoimmune skin condition causing red, scaly patches.',               'dos':['Moisturise skin daily','Use prescribed topical treatments','Manage stress — it is a trigger'],             'donts':['Avoid scratching plaques','No hot showers — use lukewarm water','Do not stop treatment abruptly'],                 'specialist':'Dermatologist',         'icon':'🩹'},
        'Impetigo':                    {'desc':'Highly contagious bacterial skin infection causing sores and crusted blisters.','dos':['Apply prescribed antibiotic ointment','Keep sores clean and covered','Wash hands frequently'],        'donts':['Do not touch or scratch sores','Avoid contact with others until healed','Do not share towels or clothes'],          'specialist':'Dermatologist',         'icon':'🦠'},
    }

    # Fetch disease info — fallback for unlisted diseases
    info = DISEASE_INFO.get(disease, {
        'desc'      : f'{disease} is the predicted condition based on your reported symptoms.',
        'dos'       : ['Consult a doctor for proper diagnosis', 'Rest adequately', 'Stay hydrated'],
        'donts'     : ['Do not self-medicate', 'Do not ignore worsening symptoms', 'Avoid stress'],
        'specialist': 'General Physician',
        'icon'      : '🩺'
    })

    # ── Rich disease info card ─────────────────────────────────
    border_color = '#d63031' if urgent else '#00b894'
    header_bg    = '#fff5f5' if urgent else '#f0fff4'
    badge_bg     = '#d63031' if urgent else '#00b894'
    badge_text   = '⚠️ URGENT — See a doctor promptly' if urgent else '✅ Non-critical — Monitor symptoms'

    # Build dos and don'ts HTML separately to avoid nested f-string issues
    dos_html   = ''.join(
        '<div style="font-size:13px;color:#2d6a4f;padding:3px 0;">• ' + item + '</div>'
        for item in info['dos']
    )
    donts_html = ''.join(
        '<div style="font-size:13px;color:#922b21;padding:3px 0;">• ' + item + '</div>'
        for item in info['donts']
    )

    card_html = (
        '<div style="border:1.5px solid ' + border_color + ';border-radius:14px;'
        'padding:1.5rem 1.8rem;margin:1rem 0;'
        'background:linear-gradient(135deg,' + header_bg + ',white);">'

        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">'
            '<span style="font-size:2.2rem;">' + info['icon'] + '</span>'
            '<div>'
                '<div style="font-size:1.25rem;font-weight:700;color:#0d1b2a;">' + disease + '</div>'
                '<span style="font-size:11px;font-weight:600;padding:3px 10px;'
                'border-radius:99px;background:' + badge_bg + ';color:white;">'
                + badge_text +
                '</span>'
            '</div>'
            '<div style="margin-left:auto;text-align:right;">'
                '<div style="font-size:11px;color:#888;">Consult a</div>'
                '<div style="font-size:13px;font-weight:600;color:' + border_color + ';">'
                + info['specialist'] +
                '</div>'
            '</div>'
        '</div>'

        '<div style="font-size:13.5px;color:#3d4d5c;line-height:1.7;'
        'padding:10px 14px;background:#f5f7fa;'
        'border-radius:8px;margin-bottom:1rem;">'
        + info['desc'] +
        '</div>'

        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">'
            '<div style="background:#f0fff4;border-radius:10px;padding:12px 16px;">'
                '<div style="font-size:12px;font-weight:600;color:#00b894;'
                'margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">'
                '✅ What to do'
                '</div>'
                + dos_html +
            '</div>'
            '<div style="background:#fff5f5;border-radius:10px;padding:12px 16px;">'
                '<div style="font-size:12px;font-weight:600;color:#d63031;'
                'margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">'
                '❌ What to avoid'
                '</div>'
                + donts_html +
            '</div>'
        '</div>'

        '<div style="font-size:11px;color:#adb5bd;margin-top:12px;padding-top:10px;'
        'border-top:1px solid #eee;">'
        '⚠️ This is an AI prediction for educational purposes only. '
        'Always consult a qualified medical professional for diagnosis and treatment.'
        '</div>'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    st.divider()
    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("#### Top 5 Possible Conditions")
        fig = px.bar(top5_df.sort_values('Probability'), x='Probability', y='Disease',
                     orientation='h', text='Probability',
                     color='Probability', color_continuous_scale=['#c8dff8','#1a5faa'])
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=300, coloraxis_showscale=False, xaxis=dict(range=[0,115]),
                          margin=dict(l=0,r=50,t=10,b=10),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Confidence Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix':'%'},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'#2979d0','thickness':0.25},
                'steps':[{'range':[0,50],'color':'#fce4e4'},
                         {'range':[50,70],'color':'#fef3e2'},
                         {'range':[70,100],'color':'#e6f4ea'}],
                'threshold':{'line':{'color':'#d63031','width':3},'thickness':0.75,'value':70}
            }
        ))
        fig_g.update_layout(height=220, margin=dict(t=30,b=10,l=10,r=10),
                             paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption("Red threshold at 70% — below this triggers an urgent flag.")

    with right:
        st.markdown("#### Symptoms You Selected")
        for s in selected:
            st.markdown(f"- {fmt_symptom(s)}")
        if selected:
            st.markdown("---")
            st.markdown("#### Symptom coverage")
            st.progress(len(selected)/len(symptom_cols))
            st.caption(f"{len(selected)} of {len(symptom_cols)} symptoms "
                       f"({round(len(selected)/len(symptom_cols)*100,1)}%)")

    st.divider()
    st.markdown("#### Why This Prediction? (SHAP Explanation)")
    st.caption("Blue = supports this diagnosis · Red = works against it")

    with st.spinner("Computing SHAP values..."):
        try:
            import shap
            from utils.shap_utils import get_shap_explanation, top_contributing_symptoms
            pred_class = model.predict([input_vec])[0]
            shap_df    = get_shap_explanation(model, input_vec, symptom_cols, pred_class)
            top_shap   = top_contributing_symptoms(shap_df, 15, present_only=True)
            if len(top_shap) == 0:
                top_shap = shap_df.head(10)
            top_shap['Symptom'] = top_shap['Symptom'].str.replace('_',' ').str.title()
            fig_s = px.bar(top_shap.sort_values('SHAP'), x='SHAP', y='Symptom', orientation='h',
                           color='SHAP', color_continuous_scale=['#e74c3c','#f8f9fa','#2979d0'],
                           color_continuous_midpoint=0, labels={'SHAP':'SHAP contribution'})
            fig_s.update_layout(height=420, margin=dict(l=0,r=20,t=10,b=10),
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP unavailable: {e}")

st.markdown("""<div class="disclaimer">
⚠️ <b>Medical Disclaimer:</b> This is an <b>educational ML project</b> and is not a substitute
for professional medical diagnosis or treatment. Always consult a qualified healthcare professional.
<br>Built with Python · Scikit-learn · XGBoost · SHAP · Streamlit · Plotly
</div>""", unsafe_allow_html=True)