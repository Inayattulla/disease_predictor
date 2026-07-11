# ============================================================
#  EDA & PREPROCESSING — disease_eda.py
#  Run this first to understand your data before modelling
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs('assets', exist_ok=True)

print("Loading data...")
train_df = pd.read_csv('Training.csv')
test_df  = pd.read_csv('Testing.csv')

train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df  = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

symptom_cols = train_df.columns[:-1].tolist()
diseases     = sorted(train_df['prognosis'].unique())

print(f"\nDataset overview")
print(f"  Training rows : {len(train_df)}")
print(f"  Test rows     : {len(test_df)}")
print(f"  Symptoms      : {len(symptom_cols)}")
print(f"  Diseases      : {len(diseases)}")
print(f"  Missing values: {train_df.isnull().sum().sum()}")
print(f"  Duplicate rows: {train_df.duplicated().sum()}")

# ── 1. Class distribution ─────────────────────────────────────
print("\nPlotting class distribution...")
fig, ax = plt.subplots(figsize=(10, 12))
counts = train_df['prognosis'].value_counts()
colors = ['#378ADD' if c == counts.max() else '#B5D4F4' for c in counts.values]
bars = ax.barh(counts.index, counts.values, color=colors, height=0.7, edgecolor='white')
ax.set_xlabel('Number of samples', fontsize=11)
ax.set_title('Disease class distribution\n(each disease has 120 training samples — perfectly balanced)', fontsize=12)
ax.invert_yaxis()
for bar in bars:
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va='center', fontsize=8)
plt.tight_layout()
plt.savefig('assets/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: assets/class_distribution.png")

# ── 2. Symptom frequency ──────────────────────────────────────
print("Plotting symptom frequency...")
freq = train_df[symptom_cols].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(30), freq.values[:30], color='#1D9E75', edgecolor='white')
ax.set_xticks(range(30))
ax.set_xticklabels([s.replace('_', ' ') for s in freq.index[:30]],
                   rotation=45, ha='right', fontsize=8)
ax.set_title('Top 30 most frequent symptoms', fontsize=12)
ax.set_ylabel('Frequency across patients')
plt.tight_layout()
plt.savefig('assets/symptom_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: assets/symptom_frequency.png")

# ── 3. Symptoms per disease heatmap ──────────────────────────
print("Plotting disease-symptom heatmap (this may take a moment)...")
disease_symptom = train_df.groupby('prognosis')[symptom_cols].mean()
top_symptoms    = train_df[symptom_cols].sum().sort_values(ascending=False).head(40).index

fig, ax = plt.subplots(figsize=(22, 14))
sns.heatmap(
    disease_symptom[top_symptoms],
    ax=ax, cmap='Blues',
    linewidths=0.3, linecolor='white',
    cbar_kws={'label': 'Symptom prevalence in disease (0–1)'}
)
ax.set_title('Disease vs Top-40 Symptom Heatmap\n(darker = symptom more common in that disease)', fontsize=13)
ax.set_xlabel('Symptom', fontsize=10)
ax.set_ylabel('Disease', fontsize=10)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('assets/disease_symptom_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: assets/disease_symptom_heatmap.png")

# ── 4. Symptoms per disease (count) ───────────────────────────
print("Plotting symptom count per disease...")
symptom_counts = train_df.groupby('prognosis')[symptom_cols].sum().sum(axis=1).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(symptom_counts.index, symptom_counts.values, color='#7F77DD', height=0.7)
ax.set_xlabel('Total symptom occurrences across all patients')
ax.set_title('Total symptom burden per disease', fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('assets/symptom_count_per_disease.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: assets/symptom_count_per_disease.png")

# ── 5. Correlation among symptoms ────────────────────────────
print("Computing symptom correlation (top 30)...")
top30 = train_df[symptom_cols].sum().sort_values(ascending=False).head(30).index
corr  = train_df[top30].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, ax=ax, mask=mask,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.3, annot=False,
    xticklabels=[s.replace('_', ' ') for s in top30],
    yticklabels=[s.replace('_', ' ') for s in top30]
)
ax.set_title('Symptom correlation matrix (top 30 symptoms)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig('assets/symptom_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: assets/symptom_correlation.png")

# ── 6. Unique symptom fingerprints ───────────────────────────
print("\nDisease uniqueness analysis:")
print("(how many symptoms are exclusive to only one disease?)\n")
symptom_disease_count = {}
for sym in symptom_cols:
    diseases_with = train_df[train_df[sym] == 1]['prognosis'].unique()
    symptom_disease_count[sym] = len(diseases_with)

exclusive  = {s: c for s, c in symptom_disease_count.items() if c == 1}
shared_few = {s: c for s, c in symptom_disease_count.items() if 2 <= c <= 5}
common     = {s: c for s, c in symptom_disease_count.items() if c > 10}

print(f"  Exclusive symptoms (appear in 1 disease only): {len(exclusive)}")
print(f"  Selective symptoms (2–5 diseases): {len(shared_few)}")
print(f"  Common symptoms (>10 diseases): {len(common)}")

if exclusive:
    print("\n  Exclusive symptom examples:")
    for s, d in list(exclusive.items())[:8]:
        disease_name = train_df[train_df[s] == 1]['prognosis'].unique()[0]
        print(f"    '{s.replace('_', ' ')}' → only in: {disease_name}")

print("\n✓ EDA complete. All charts saved to assets/")
print("  Next: run disease_prediction_model.py to train the model.")