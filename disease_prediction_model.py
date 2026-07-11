# ============================================================
#  disease_prediction_model.py
#  Full ML training pipeline — Random Forest + XGBoost + SHAP
#  Run this FIRST before launching the Streamlit app.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap

from config import (
    TRAIN_CSV, TEST_CSV, MODEL_PATH, XGB_MODEL_PATH,
    ENCODER_PATH, SYMPTOM_PATH, ASSETS_DIR,
    RF_PARAMS, XGB_PARAMS, TEST_SIZE, RANDOM_STATE, CV_FOLDS
)

# Create output dirs
os.makedirs('models', exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── STEP 1: Load data ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df  = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

symptom_cols = train_df.columns[:-1].tolist()
print(f"Train: {train_df.shape} | Test: {test_df.shape}")
print(f"Diseases: {train_df['prognosis'].nunique()} | Symptoms: {len(symptom_cols)}")
print(f"Missing values: {train_df.isnull().sum().sum()}")

# ── STEP 2: Prepare data ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Encoding labels + splitting data")
print("=" * 60)

X_train = train_df.drop('prognosis', axis=1)
y_train = train_df['prognosis']
X_test  = test_df.drop('prognosis', axis=1)[X_train.columns]
y_test  = test_df['prognosis']

le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

X_tr = X_train.values
X_te = X_test.values
print(f"X_train: {X_tr.shape} | X_test: {X_te.shape}")
print(f"Classes: {len(le.classes_)}")

# ── STEP 3: Train models ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training models")
print("=" * 60)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X_tr, y_train_enc)
rf_acc = accuracy_score(y_test_enc, rf.predict(X_te))
rf_cv  = cross_val_score(rf, X_tr, y_train_enc, cv=cv, scoring='accuracy')
print(f"  Test accuracy : {rf_acc*100:.2f}%")
print(f"  CV accuracy   : {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")

# XGBoost
print("\nTraining XGBoost...")
xgb = XGBClassifier(**XGB_PARAMS)
xgb.fit(X_tr, y_train_enc, eval_set=[(X_te, y_test_enc)], verbose=False)
xgb_acc = accuracy_score(y_test_enc, xgb.predict(X_te))
xgb_cv  = cross_val_score(xgb, X_tr, y_train_enc, cv=cv, scoring='accuracy')
print(f"  Test accuracy : {xgb_acc*100:.2f}%")
print(f"  CV accuracy   : {xgb_cv.mean()*100:.2f}% ± {xgb_cv.std()*100:.2f}%")

# Naive Bayes (baseline)
print("\nTraining Naive Bayes (baseline)...")
nb = GaussianNB()
nb.fit(X_tr, y_train_enc)
nb_acc = accuracy_score(y_test_enc, nb.predict(X_te))
print(f"  Test accuracy : {nb_acc*100:.2f}%")

# ── STEP 4: Evaluation ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Detailed evaluation")
print("=" * 60)

best = rf
best_pred = best.predict(X_te)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test_enc, best_pred, target_names=le.classes_))

# Model comparison chart
fig, ax = plt.subplots(figsize=(8, 4))
names  = ['Naive Bayes\n(baseline)', 'XGBoost', 'Random Forest']
scores = [nb_acc, xgb_acc, rf_acc]
colors = ['#B4B2A9', '#5DCAA5', '#378ADD']
bars   = ax.barh(names, [s*100 for s in scores], color=colors, height=0.5)
ax.set_xlim(80, 102)
ax.set_xlabel('Test Accuracy (%)')
ax.set_title('Model Comparison')
for bar, score in zip(bars, scores):
    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
            f'{score*100:.1f}%', va='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{ASSETS_DIR}model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {ASSETS_DIR}model_comparison.png")

# Confusion matrix
fig, ax = plt.subplots(figsize=(18, 16))
cm = confusion_matrix(y_test_enc, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, linewidths=0.3)
ax.set_title('Confusion Matrix — Random Forest', fontsize=14)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(f'{ASSETS_DIR}confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {ASSETS_DIR}confusion_matrix.png")

# ── STEP 5: Feature importance ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature importance")
print("=" * 60)

feat_imp = pd.Series(best.feature_importances_, index=symptom_cols).sort_values(ascending=False)
print("Top 10 symptoms:")
print(feat_imp.head(10).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
feat_imp.head(20).sort_values().plot(kind='barh', ax=ax, color='#378ADD')
ax.set_title('Top 20 Feature Importances (Random Forest)')
ax.set_xlabel('Importance score')
plt.tight_layout()
plt.savefig(f'{ASSETS_DIR}feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {ASSETS_DIR}feature_importance.png")

# ── STEP 6: SHAP ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: SHAP explainability")
print("=" * 60)

print("Computing SHAP values (may take ~30 seconds)...")
explainer   = shap.TreeExplainer(best)
X_explain   = X_te[:80]
shap_values = explainer.shap_values(X_explain)

shap.summary_plot(shap_values, X_explain, feature_names=symptom_cols,
                  plot_type='bar', max_display=20, show=False)
plt.title('Global SHAP — Top 20 Symptoms')
plt.tight_layout()
plt.savefig(f'{ASSETS_DIR}shap_global.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {ASSETS_DIR}shap_global.png")

# ── STEP 7: Save artifacts ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving model artifacts")
print("=" * 60)

joblib.dump(best,          MODEL_PATH)
joblib.dump(xgb,           XGB_MODEL_PATH)
joblib.dump(le,            ENCODER_PATH)
joblib.dump(symptom_cols,  SYMPTOM_PATH)

print(f"Saved: {MODEL_PATH}")
print(f"Saved: {XGB_MODEL_PATH}")
print(f"Saved: {ENCODER_PATH}")
print(f"Saved: {SYMPTOM_PATH}")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best model    : Random Forest")
print(f"  Test accuracy : {rf_acc*100:.2f}%")
print(f"  CV accuracy   : {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")
print(f"  XGBoost acc   : {xgb_acc*100:.2f}%")
print(f"  Naive Bayes   : {nb_acc*100:.2f}% (baseline)")
print()
print("  Next step: streamlit run app.py")
print("=" * 60)