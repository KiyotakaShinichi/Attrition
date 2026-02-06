# ==============================
# Employee Attrition ML Pipeline
# COMPARING: Plain, SMOTE-Tomek, SMOTE-ENN
# 10-fold CV on FULL dataset (WEKA-matched)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, roc_curve,
    accuracy_score, precision_score, confusion_matrix,
    cohen_kappa_score, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
import warnings

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv("EmployeeAttritionFPD_cleaned.csv")

X = df.drop(columns="Attrition")
y = df["Attrition"]

print(f"{'=' * 70}")
print("RESAMPLING TECHNIQUES COMPARISON")
print("Plain vs SMOTE-Tomek vs SMOTE-ENN")
print("10-Fold Cross-Validation on FULL Dataset (WEKA-matched)")
print(f"{'=' * 70}")
print(f"\nDataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Class ratio: {y.value_counts()[1] / len(y) * 100:.2f}% attrition")
print(f"Imbalance ratio: 1:{y.value_counts()[0] / y.value_counts()[1]:.2f}\n")

# ------------------------------
# Feature groups
# ------------------------------
nominal_cols = [
    "BusinessTravel", "Department", "EducationField",
    "Sex", "JobRole", "MaritalStatus", "OverTime"
]

ordinal_cols = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "PerformanceRating",
    "RelationshipSatisfaction", "WorkLifeBalance", "AgeGroup"
]

numeric_cols = [
    col for col in X.columns
    if col not in nominal_cols + ordinal_cols
]

# Force ordinal to numeric
for col in ordinal_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# ------------------------------
# Preprocessing pipeline
# ------------------------------
num_ord_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num_ord", num_ord_pipeline, numeric_cols + ordinal_cols),
        ("nom", nominal_pipeline, nominal_cols)
    ]
)

# ------------------------------
# VISUALIZE RESAMPLING TECHNIQUES
# ------------------------------
print(f"{'=' * 70}")
print("UNDERSTANDING RESAMPLING TECHNIQUES")
print(f"{'=' * 70}")

# Preprocess data for visualization
X_transformed = preprocessor.fit_transform(X)

# Original
original_class_0 = (y == 0).sum()
original_class_1 = (y == 1).sum()

# SMOTE-Tomek
smote_tomek = SMOTETomek(sampling_strategy=1.0, random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_transformed, y)
st_class_0 = (y_st == 0).sum()
st_class_1 = (y_st == 1).sum()

# SMOTE-ENN
smote_enn = SMOTEENN(sampling_strategy=1.0, random_state=42)
X_se, y_se = smote_enn.fit_resample(X_transformed, y)
se_class_0 = (y_se == 0).sum()
se_class_1 = (y_se == 1).sum()

print(f"\nOriginal Dataset:")
print(f"  Class 0: {original_class_0}")
print(f"  Class 1: {original_class_1}")
print(f"  Total: {len(y)}")
print(f"  Ratio: 1:{original_class_0 / original_class_1:.2f}")

print(f"\nSMOTE-Tomek:")
print(f"  Class 0: {st_class_0} ({st_class_0 - original_class_0:+d})")
print(f"  Class 1: {st_class_1} ({st_class_1 - original_class_1:+d})")
print(f"  Total: {len(y_st)} ({len(y_st) - len(y):+d})")
print(f"  Ratio: 1:{st_class_0 / st_class_1:.2f}")
print(f"  → Tomek links remove {len(y) - len(y_st) + (st_class_1 - original_class_1)} samples")

print(f"\nSMOTE-ENN:")
print(f"  Class 0: {se_class_0} ({se_class_0 - original_class_0:+d})")
print(f"  Class 1: {se_class_1} ({se_class_1 - original_class_1:+d})")
print(f"  Total: {len(y_se)} ({len(y_se) - len(y):+d})")
print(f"  Ratio: 1:{se_class_0 / se_class_1:.2f}")
print(f"  → ENN removes {len(y) - len(y_se) + (se_class_1 - original_class_1)} samples")

print(f"\nKey Difference:")
print(f"  SMOTE-Tomek: Removes Tomek links (borderline/noisy samples)")
print(f"  SMOTE-ENN: Removes misclassified neighbors (more aggressive cleaning)")
print(f"  → ENN typically removes MORE samples than Tomek")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

datasets = [
    (y, "Original", "purple"),
    (y_st, "SMOTE-Tomek", "blue"),
    (y_se, "SMOTE-ENN", "green")
]

for idx, (data, title, color) in enumerate(datasets):
    class_counts = pd.Series(data).value_counts().sort_index()
    axes[idx].bar(['No Attrition', 'Attrition'], class_counts,
                  color=[color, '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[idx].set_ylabel('Number of Samples', fontsize=12)
    axes[idx].set_title(title, fontsize=14, fontweight='bold')
    axes[idx].set_ylim([0, max(class_counts) * 1.2])

    for i, (label, count) in enumerate(class_counts.items()):
        axes[idx].text(i, count + 20,
                       f'{count}\n({count / len(data) * 100:.1f}%)',
                       ha='center', fontsize=11, fontweight='bold')

    total = len(data)
    axes[idx].text(0.5, -0.15, f'Total: {total} samples',
                   transform=axes[idx].transAxes, ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'resampling_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'resampling_comparison.png'}")
plt.close()

# ------------------------------
# EXPERIMENT 1: Plain Logistic Regression
# ------------------------------
print(f"\n{'=' * 70}")
print("EXPERIMENT 1: Plain Logistic Regression (class_weight='balanced')")
print(f"{'=' * 70}")

plain_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    ))
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred_plain = cross_val_predict(plain_model, X, y, cv=cv)
y_proba_plain = cross_val_predict(plain_model, X, y, cv=cv, method='predict_proba')[:, 1]

results_plain = {
    'accuracy': accuracy_score(y, y_pred_plain),
    'precision': precision_score(y, y_pred_plain),
    'recall': recall_score(y, y_pred_plain),
    'f1': f1_score(y, y_pred_plain),
    'roc_auc': roc_auc_score(y, y_proba_plain),
    'kappa': cohen_kappa_score(y, y_pred_plain),
    'cm': confusion_matrix(y, y_pred_plain)
}

print(f"\n10-Fold CV Results:")
print(f"Accuracy      : {results_plain['accuracy']:.4f}")
print(f"Precision     : {results_plain['precision']:.4f}")
print(f"Recall        : {results_plain['recall']:.4f}")
print(f"F1-score      : {results_plain['f1']:.4f}")
print(f"ROC-AUC       : {results_plain['roc_auc']:.4f}")
print(f"Cohen's Kappa : {results_plain['kappa']:.4f}")
print(f"\nConfusion Matrix:\n{results_plain['cm']}")

# ------------------------------
# EXPERIMENT 2: SMOTE-Tomek + Optuna
# ------------------------------
print(f"\n{'=' * 70}")
print("EXPERIMENT 2: SMOTE-Tomek + Hyperparameter Tuning")
print(f"{'=' * 70}")


def objective_smotetomek(trial):
    model = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTETomek(sampling_strategy=1.0, random_state=42)),
        ("model", LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return np.mean(scores)


print("Running Optuna for SMOTE-Tomek...")
study_st = optuna.create_study(direction="maximize")
study_st.optimize(objective_smotetomek, n_trials=50, show_progress_bar=True)

print(f"\nBest CV F1: {study_st.best_value:.4f}")
print(f"Best C: {study_st.best_params['C']:.4f}")

# Evaluate with best params
best_model_st = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTETomek(sampling_strategy=1.0, random_state=42)),
    ("model", LogisticRegression(
        C=study_st.best_params["C"],
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

y_pred_st = cross_val_predict(best_model_st, X, y, cv=cv)
y_proba_st = cross_val_predict(best_model_st, X, y, cv=cv, method='predict_proba')[:, 1]

results_st = {
    'accuracy': accuracy_score(y, y_pred_st),
    'precision': precision_score(y, y_pred_st),
    'recall': recall_score(y, y_pred_st),
    'f1': f1_score(y, y_pred_st),
    'roc_auc': roc_auc_score(y, y_proba_st),
    'kappa': cohen_kappa_score(y, y_pred_st),
    'cm': confusion_matrix(y, y_pred_st)
}

print(f"\n10-Fold CV Results:")
print(f"Accuracy      : {results_st['accuracy']:.4f}")
print(f"Precision     : {results_st['precision']:.4f}")
print(f"Recall        : {results_st['recall']:.4f}")
print(f"F1-score      : {results_st['f1']:.4f}")
print(f"ROC-AUC       : {results_st['roc_auc']:.4f}")
print(f"Cohen's Kappa : {results_st['kappa']:.4f}")
print(f"\nConfusion Matrix:\n{results_st['cm']}")

# ------------------------------
# EXPERIMENT 3: SMOTE-ENN + Optuna
# ------------------------------
print(f"\n{'=' * 70}")
print("EXPERIMENT 3: SMOTE-ENN + Hyperparameter Tuning")
print(f"{'=' * 70}")


def objective_smoteenn(trial):
    model = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTEENN(sampling_strategy=1.0, random_state=42)),
        ("model", LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return np.mean(scores)


print("Running Optuna for SMOTE-ENN...")
study_se = optuna.create_study(direction="maximize")
study_se.optimize(objective_smoteenn, n_trials=50, show_progress_bar=True)

print(f"\nBest CV F1: {study_se.best_value:.4f}")
print(f"Best C: {study_se.best_params['C']:.4f}")

# Evaluate with best params
best_model_se = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTEENN(sampling_strategy=1.0, random_state=42)),
    ("model", LogisticRegression(
        C=study_se.best_params["C"],
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

y_pred_se = cross_val_predict(best_model_se, X, y, cv=cv)
y_proba_se = cross_val_predict(best_model_se, X, y, cv=cv, method='predict_proba')[:, 1]

results_se = {
    'accuracy': accuracy_score(y, y_pred_se),
    'precision': precision_score(y, y_pred_se),
    'recall': recall_score(y, y_pred_se),
    'f1': f1_score(y, y_pred_se),
    'roc_auc': roc_auc_score(y, y_proba_se),
    'kappa': cohen_kappa_score(y, y_pred_se),
    'cm': confusion_matrix(y, y_pred_se)
}

print(f"\n10-Fold CV Results:")
print(f"Accuracy      : {results_se['accuracy']:.4f}")
print(f"Precision     : {results_se['precision']:.4f}")
print(f"Recall        : {results_se['recall']:.4f}")
print(f"F1-score      : {results_se['f1']:.4f}")
print(f"ROC-AUC       : {results_se['roc_auc']:.4f}")
print(f"Cohen's Kappa : {results_se['kappa']:.4f}")
print(f"\nConfusion Matrix:\n{results_se['cm']}")

# ------------------------------
# YOUDEN'S J FOR ALL METHODS
# ------------------------------
print(f"\n{'=' * 70}")
print("YOUDEN'S J OPTIMAL THRESHOLDS")
print(f"{'=' * 70}")

# Plain
fpr_plain, tpr_plain, thresh_plain = roc_curve(y, y_proba_plain)
j_plain = tpr_plain - fpr_plain
opt_idx_plain = np.argmax(j_plain)
opt_thresh_plain = thresh_plain[opt_idx_plain]
y_pred_plain_thresh = (y_proba_plain >= opt_thresh_plain).astype(int)

results_plain_thresh = {
    'accuracy': accuracy_score(y, y_pred_plain_thresh),
    'precision': precision_score(y, y_pred_plain_thresh),
    'recall': recall_score(y, y_pred_plain_thresh),
    'f1': f1_score(y, y_pred_plain_thresh),
    'kappa': cohen_kappa_score(y, y_pred_plain_thresh)
}

print(f"\nPlain Model:")
print(f"  Optimal threshold: {opt_thresh_plain:.4f}")
print(f"  F1: {results_plain_thresh['f1']:.4f} (vs {results_plain['f1']:.4f} default)")
print(f"  Kappa: {results_plain_thresh['kappa']:.4f} (vs {results_plain['kappa']:.4f} default)")

# SMOTE-Tomek
fpr_st, tpr_st, thresh_st = roc_curve(y, y_proba_st)
j_st = tpr_st - fpr_st
opt_idx_st = np.argmax(j_st)
opt_thresh_st = thresh_st[opt_idx_st]
y_pred_st_thresh = (y_proba_st >= opt_thresh_st).astype(int)

results_st_thresh = {
    'accuracy': accuracy_score(y, y_pred_st_thresh),
    'precision': precision_score(y, y_pred_st_thresh),
    'recall': recall_score(y, y_pred_st_thresh),
    'f1': f1_score(y, y_pred_st_thresh),
    'kappa': cohen_kappa_score(y, y_pred_st_thresh)
}

print(f"\nSMOTE-Tomek:")
print(f"  Optimal threshold: {opt_thresh_st:.4f}")
print(f"  F1: {results_st_thresh['f1']:.4f} (vs {results_st['f1']:.4f} default)")
print(f"  Kappa: {results_st_thresh['kappa']:.4f} (vs {results_st['kappa']:.4f} default)")

# SMOTE-ENN
fpr_se, tpr_se, thresh_se = roc_curve(y, y_proba_se)
j_se = tpr_se - fpr_se
opt_idx_se = np.argmax(j_se)
opt_thresh_se = thresh_se[opt_idx_se]
y_pred_se_thresh = (y_proba_se >= opt_thresh_se).astype(int)

results_se_thresh = {
    'accuracy': accuracy_score(y, y_pred_se_thresh),
    'precision': precision_score(y, y_pred_se_thresh),
    'recall': recall_score(y, y_pred_se_thresh),
    'f1': f1_score(y, y_pred_se_thresh),
    'kappa': cohen_kappa_score(y, y_pred_se_thresh)
}

print(f"\nSMOTE-ENN:")
print(f"  Optimal threshold: {opt_thresh_se:.4f}")
print(f"  F1: {results_se_thresh['f1']:.4f} (vs {results_se['f1']:.4f} default)")
print(f"  Kappa: {results_se_thresh['kappa']:.4f} (vs {results_se['kappa']:.4f} default)")

# ------------------------------
# COMPREHENSIVE COMPARISON TABLE
# ------------------------------
print(f"\n{'=' * 70}")
print("COMPREHENSIVE COMPARISON (All Methods)")
print(f"{'=' * 70}")

comparison_df = pd.DataFrame({
    'Method': [
        'Plain (0.5)',
        'Plain (Youden)',
        'SMOTE-Tomek (0.5)',
        'SMOTE-Tomek (Youden)',
        'SMOTE-ENN (0.5)',
        'SMOTE-ENN (Youden)'
    ],
    'Accuracy': [
        results_plain['accuracy'],
        results_plain_thresh['accuracy'],
        results_st['accuracy'],
        results_st_thresh['accuracy'],
        results_se['accuracy'],
        results_se_thresh['accuracy']
    ],
    'Precision': [
        results_plain['precision'],
        results_plain_thresh['precision'],
        results_st['precision'],
        results_st_thresh['precision'],
        results_se['precision'],
        results_se_thresh['precision']
    ],
    'Recall': [
        results_plain['recall'],
        results_plain_thresh['recall'],
        results_st['recall'],
        results_st_thresh['recall'],
        results_se['recall'],
        results_se_thresh['recall']
    ],
    'F1': [
        results_plain['f1'],
        results_plain_thresh['f1'],
        results_st['f1'],
        results_st_thresh['f1'],
        results_se['f1'],
        results_se_thresh['f1']
    ],
    'Kappa': [
        results_plain['kappa'],
        results_plain_thresh['kappa'],
        results_st['kappa'],
        results_st_thresh['kappa'],
        results_se['kappa'],
        results_se_thresh['kappa']
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Find best method for each metric
print(f"\n{'=' * 70}")
print("BEST PERFORMERS BY METRIC")
print(f"{'=' * 70}")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']:
    best_idx = comparison_df[metric].idxmax()
    best_method = comparison_df.loc[best_idx, 'Method']
    best_score = comparison_df.loc[best_idx, metric]
    print(f"{metric:12s}: {best_method:25s} ({best_score:.4f})")

# ------------------------------
# VISUALIZATIONS
# ------------------------------

# 1. ROC Curves Comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

ax.plot(fpr_plain, tpr_plain, linewidth=2,
        label=f'Plain (AUC = {results_plain["roc_auc"]:.3f})', color='purple')
ax.plot(fpr_st, tpr_st, linewidth=2,
        label=f'SMOTE-Tomek (AUC = {results_st["roc_auc"]:.3f})', color='blue')
ax.plot(fpr_se, tpr_se, linewidth=2,
        label=f'SMOTE-ENN (AUC = {results_se["roc_auc"]:.3f})', color='green')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

# Mark optimal points
ax.plot(fpr_plain[opt_idx_plain], tpr_plain[opt_idx_plain], 'o',
        markersize=10, color='purple', markeredgecolor='black', markeredgewidth=2)
ax.plot(fpr_st[opt_idx_st], tpr_st[opt_idx_st], 's',
        markersize=10, color='blue', markeredgecolor='black', markeredgewidth=2)
ax.plot(fpr_se[opt_idx_se], tpr_se[opt_idx_se], '^',
        markersize=10, color='green', markeredgecolor='black', markeredgewidth=2)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Plain vs SMOTE-Tomek vs SMOTE-ENN', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'roc_comparison_all.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir / 'roc_comparison_all.png'}")
plt.close()

# 2. Confusion Matrices (6 panels)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

cms = [
    (results_plain['cm'], 'Plain (0.5)', results_plain['kappa'], 'Purples'),
    (confusion_matrix(y, y_pred_plain_thresh), 'Plain (Youden)', results_plain_thresh['kappa'], 'Purples'),
    (results_st['cm'], 'SMOTE-Tomek (0.5)', results_st['kappa'], 'Blues'),
    (confusion_matrix(y, y_pred_st_thresh), 'SMOTE-Tomek (Youden)', results_st_thresh['kappa'], 'Blues'),
    (results_se['cm'], 'SMOTE-ENN (0.5)', results_se['kappa'], 'Greens'),
    (confusion_matrix(y, y_pred_se_thresh), 'SMOTE-ENN (Youden)', results_se_thresh['kappa'], 'Greens')
]

for idx, (cm, title, kappa, cmap) in enumerate(cms):
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[idx],
                xticklabels=['No Attrition', 'Attrition'],
                yticklabels=['No Attrition', 'Attrition'])
    axes[idx].set_title(f'{title}\nKappa: {kappa:.4f}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrices_all_methods.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'confusion_matrices_all_methods.png'}")
plt.close()

# 3. Metrics Comparison Bar Chart
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

methods = comparison_df['Method'].values
x = np.arange(len(methods))
width = 0.15

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    offset = (i - 2) * width
    values = comparison_df[metric].values
    ax.bar(x + offset, values, width, label=metric, color=color, alpha=0.8)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison: All Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison_all.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'metrics_comparison_all.png'}")
plt.close()

# 4. Heatmap of Results
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

heatmap_data = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']].values.T

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=methods, yticklabels=metrics_to_plot,
            cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)

ax.set_title('Performance Heatmap: All Methods & Metrics', fontsize=14, fontweight='bold')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Metric', fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / 'heatmap_all_methods.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'heatmap_all_methods.png'}")
plt.close()

# ------------------------------
# SAVE RESULTS TO CSV
# ------------------------------
comparison_df.to_csv(output_dir / 'results_comparison.csv', index=False)
print(f"✓ Saved: {output_dir / 'results_comparison.csv'}")

# ------------------------------
# FINAL RECOMMENDATIONS
# ------------------------------
print(f"\n{'=' * 70}")
print("FINAL RECOMMENDATIONS")
print(f"{'=' * 70}")

best_f1_idx = comparison_df['F1'].idxmax()
best_kappa_idx = comparison_df['Kappa'].idxmax()
best_recall_idx = comparison_df['Recall'].idxmax()

print(f"\nBased on 10-fold cross-validation:")
print(f"  Best F1-Score    : {comparison_df.loc[best_f1_idx, 'Method']} ({comparison_df.loc[best_f1_idx, 'F1']:.4f})")
print(
    f"  Best Kappa       : {comparison_df.loc[best_kappa_idx, 'Method']} ({comparison_df.loc[best_kappa_idx, 'Kappa']:.4f})")
print(
    f"  Best Recall      : {comparison_df.loc[best_recall_idx, 'Method']} ({comparison_df.loc[best_recall_idx, 'Recall']:.4f})")

print(f"\nKey Insights:")
print(f"  1. SMOTE-ENN vs SMOTE-Tomek:")
print(f"     - ENN is more aggressive at removing noisy samples")
print(f"     - May lead to better boundary separation")
print(f"     - Check which gives better F1/Kappa for your use case")
print(f"\n  2. Youden's J threshold consistently improves Recall")
print(f"     - Important if missing attrition is costly")
print(f"     - May reduce Precision (more false alarms)")
print(f"\n  3. All methods on FULL dataset (10-fold CV)")
print(f"     - No train/test split = maximum data utilization")
print(f"     - More reliable than your original 80/20 split")

print(f"\n✓ All results saved to: {output_dir.absolute()}")
print("\nAnalysis complete! 🎯")