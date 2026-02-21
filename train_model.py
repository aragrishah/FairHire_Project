import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = 'synthetic_hiring_data.csv'
OUTPUT_DIR  = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("⏳ Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Hired distribution:\n{df['hired'].value_counts(normalize=True).round(3)}\n")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("🔄 Preprocessing...")

# Drop non-informative columns
df = df.drop(columns=['candidate_id', 'age_group'], errors='ignore')

# Fill any missing values
df = df.fillna(0)

# Encode categoricals
categorical_cols = [
    'gender', 'nationality', 'education_level', 'field_of_study',
    'university_tier', 'job_role_applied', 'application_source'
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

pickle.dump(encoders, open(os.path.join(OUTPUT_DIR, 'encoders.pkl'), 'wb'))
print(f"   Encoders saved for {len(encoders)} categorical columns.\n")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (stratified to preserve class ratio)
# ─────────────────────────────────────────────
target = 'hired'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Split: {len(X_train)} train / {len(X_test)} test")
print(f"   Train hired rate : {y_train.mean():.2%}")
print(f"   Test  hired rate : {y_test.mean():.2%}\n")

# ─────────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE  (class_weight='balanced')
# ─────────────────────────────────────────────
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(zip(np.unique(y_train), class_weights))
print(f"⚖️  Class weights: {cw_dict}\n")

# ─────────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────────
print("💪 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',   # handles imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("   Training complete.\n")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("📈 Evaluating model...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# --- Classification Report ---
print("\n── Classification Report ──────────────────────")
print(classification_report(y_test, y_pred, target_names=['Not Hired', 'Hired']))

# --- ROC-AUC ---
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score : {roc_auc:.4f}")

# --- Cross-validation (5-fold, stratified) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
print("\n🎨 Generating evaluation plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('XAI-Driven Hiring Model — Evaluation', fontsize=15, fontweight='bold')

# --- (a) Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Hired', 'Hired'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

# --- (b) ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.4f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')

# --- (c) Feature Importance ---
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.tail(15).plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_title('Top 15 Feature Importances')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: outputs/model_evaluation.png")

# ─────────────────────────────────────────────
# 8. SHAP — XAI EXPLANATIONS
# ─────────────────────────────────────────────
print("\n🔍 Computing SHAP values (this may take ~30 seconds)...")

# Use a sample for speed
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary, use class=1 (Hired) SHAP values
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# --- SHAP Summary Plot (Beeswarm) ---
fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_sample, show=False, plot_size=None)
plt.title('SHAP Feature Impact — Hiring Decision', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: outputs/shap_summary.png")

# --- SHAP Bar Plot (Mean Absolute) ---
shap.summary_plot(sv, X_sample, plot_type='bar', show=False)
plt.title('SHAP Mean Absolute Feature Importance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: outputs/shap_bar.png")

# ─────────────────────────────────────────────
# 9. BIAS / FAIRNESS AUDIT  (hiring rate by demographic)
# ─────────────────────────────────────────────
print("\n⚖️  Running Bias Audit...")

# Decode gender and nationality back for analysis
df_audit = df.copy()
df_audit['gender_label'] = encoders['gender'].inverse_transform(df_audit['gender'])
df_audit['nationality_label'] = encoders['nationality'].inverse_transform(df_audit['nationality'])

# Add model predictions
df_audit['pred_hired'] = model.predict(X)

fig_bias, axes_b = plt.subplots(1, 2, figsize=(14, 5))
fig_bias.suptitle('Hiring Rate Bias Audit', fontsize=13, fontweight='bold')

for ax, col, label in zip(
    axes_b,
    ['gender_label', 'nationality_label'],
    ['Gender', 'Nationality']
):
    audit = df_audit.groupby(col).agg(
        actual_hire_rate=('hired', 'mean'),
        predicted_hire_rate=('pred_hired', 'mean')
    ).sort_values('actual_hire_rate', ascending=False)

    x = np.arange(len(audit))
    width = 0.35
    ax.bar(x - width/2, audit['actual_hire_rate'], width, label='Actual',    color='steelblue')
    ax.bar(x + width/2, audit['predicted_hire_rate'], width, label='Predicted', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(audit.index, rotation=20, ha='right')
    ax.set_ylabel('Hire Rate')
    ax.set_title(f'Hire Rate by {label}')
    ax.legend()
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bias_audit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: outputs/bias_audit.png")

# ─────────────────────────────────────────────
# 10. SAVE MODEL & PROCESSED DATA
# ─────────────────────────────────────────────
print("\n💾 Saving final artifacts...")
pickle.dump(model, open(os.path.join(OUTPUT_DIR, 'hiring_model.pkl'), 'wb'))
df.to_csv(os.path.join(OUTPUT_DIR, 'processed_candidates.csv'), index=False)

# Save SHAP values for dashboard use
np.save(os.path.join(OUTPUT_DIR, 'shap_values.npy'), sv)
X_sample.to_csv(os.path.join(OUTPUT_DIR, 'shap_sample.csv'), index=False)

print("""
✅ Phase 1 Complete! All artifacts saved to outputs/:
   • hiring_model.pkl        — trained model
   • encoders.pkl            — label encoders
   • processed_candidates.csv
   • model_evaluation.png    — confusion matrix, ROC curve, feature importance
   • shap_summary.png        — SHAP beeswarm plot
   • shap_bar.png            — SHAP mean importance bar chart
   • bias_audit.png          — fairness audit by gender & nationality
   • shap_values.npy         — raw SHAP values (for dashboard)
   • shap_sample.csv         — sample used for SHAP
""")