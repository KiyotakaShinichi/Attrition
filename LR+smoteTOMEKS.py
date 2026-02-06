# ==============================
# Employee Attrition ML Pipeline
# Logistic Regression + Optuna
# WITH: Youden's J, Lift Charts, Learning Curves
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
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

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv("EmployeeAttritionFPD_cleaned.csv")

X = df.drop(columns="Attrition")
y = df["Attrition"]

print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Class ratio: {y.value_counts()[1] / y.value_counts()[0]:.2%}\n")

# ------------------------------
# Feature groups
# ------------------------------
nominal_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Sex",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

ordinal_cols = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "AgeGroup"
]

numeric_cols = [
    col for col in X.columns
    if col not in nominal_cols + ordinal_cols
]

# ------------------------------
# Force ordinal to numeric
# ------------------------------
for col in ordinal_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# ------------------------------
# Train / test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Test set class distribution:\n{y_test.value_counts()}\n")

# ------------------------------
# Pipelines per feature type
# ------------------------------
num_ord_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

# ------------------------------
# ColumnTransformer
# ------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num_ord", num_ord_pipeline, numeric_cols + ordinal_cols),
        ("nom", nominal_pipeline, nominal_cols)
    ]
)


# ------------------------------
# Optuna objective
# ------------------------------
def objective(trial):
    model = LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 10.0, log=True),
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # Added class weights
    )

    pipeline = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTETomek(sampling_strategy=1.0, random_state=42)),  # Full balance
        ("model", model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)

        f1_scores.append(f1_score(y_val, preds))

    return np.mean(f1_scores)


# ------------------------------
# Run Optuna
# ------------------------------
print("Running Optuna hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest CV F1: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}\n")

# ------------------------------
# Train final model
# ------------------------------
print("Training final model with best parameters...")
best_model = LogisticRegression(
    C=study.best_params["C"],
    penalty="l2",
    solver="liblinear",
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

final_pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTETomek(sampling_strategy=1.0, random_state=42)),
    ("model", best_model)
])

final_pipeline.fit(X_train, y_train)

# ------------------------------
# Get predictions and probabilities
# ------------------------------
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

# ------------------------------
# YOUDEN'S J OPTIMAL THRESHOLD
# ------------------------------
print("=" * 60)
print("FINDING OPTIMAL THRESHOLD USING YOUDEN'S J")
print("=" * 60)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Youden's J statistic = Sensitivity + Specificity - 1
youdens_j = tpr - fpr
optimal_idx = np.argmax(youdens_j)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.4f}")
print(f"Youden's J value: {youdens_j[optimal_idx]:.4f}")
print(f"Sensitivity at optimal: {tpr[optimal_idx]:.4f}")
print(f"Specificity at optimal: {1 - fpr[optimal_idx]:.4f}")

# Predictions with optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# Also get default threshold predictions for comparison
y_pred_default = (y_proba >= 0.5).astype(int)


# ------------------------------
# Performance Comparison
# ------------------------------
def print_metrics(y_true, y_pred, y_prob, title):
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc:.4f}")
    print(f"Cohen's Kappa : {kappa:.4f}")

    print(f"\nConfusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity         : {specificity:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': roc, 'kappa': kappa,
        'sensitivity': sensitivity, 'specificity': specificity
    }


# Print both results
metrics_default = print_metrics(y_test, y_pred_default, y_proba,
                                "DEFAULT THRESHOLD (0.5)")
metrics_optimal = print_metrics(y_test, y_pred_optimal, y_proba,
                                "OPTIMAL THRESHOLD (YOUDEN'S J)")

# ------------------------------
# VISUALIZATION 1: ROC Curve with Youden's J
# ------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12,
        label=f"Optimal Threshold = {optimal_threshold:.3f}\n(Youden's J = {youdens_j[optimal_idx]:.3f})")

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title("ROC Curve with Youden's J Optimal Threshold", fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/roc_curve_youdens_j.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: roc_curve_youdens_j.png")

# ------------------------------
# VISUALIZATION 2: Lift Chart & Cumulative Gains
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Sort by predicted probability (descending)
sorted_indices = np.argsort(y_proba)[::-1]
y_sorted = y_test.values[sorted_indices]

# Calculate cumulative gains
total_positives = y_sorted.sum()
cumulative_positives = np.cumsum(y_sorted)
cumulative_percent = np.arange(1, len(y_sorted) + 1) / len(y_sorted) * 100
gains = cumulative_positives / total_positives * 100

# Perfect model (all positives captured first)
perfect_gains = np.minimum(cumulative_percent * total_positives / len(y_sorted), 100)

# Plot 1: Cumulative Gains Chart
ax1.plot(cumulative_percent, gains, linewidth=2, label='Model')
ax1.plot(cumulative_percent, cumulative_percent, 'k--', linewidth=1, label='Random')
ax1.plot(cumulative_percent, perfect_gains, 'g--', linewidth=1, label='Perfect')
ax1.set_xlabel('% of Test Set (sorted by predicted probability)', fontsize=12)
ax1.set_ylabel('% of Attrition Cases Captured', fontsize=12)
ax1.set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([0, 100])

# Add annotation for top 20%
top_20_idx = int(len(y_sorted) * 0.2)
top_20_gain = gains[top_20_idx]
ax1.axvline(20, color='red', linestyle=':', alpha=0.5)
ax1.axhline(top_20_gain, color='red', linestyle=':', alpha=0.5)
ax1.text(22, top_20_gain, f'{top_20_gain:.1f}% captured\nin top 20%',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Lift Chart
baseline = total_positives / len(y_sorted) * 100
lift = gains / cumulative_percent

ax2.plot(cumulative_percent, lift, linewidth=2, label='Lift Curve', color='blue')
ax2.axhline(1, color='black', linestyle='--', linewidth=1, label='Baseline (Random)')
ax2.set_xlabel('% of Test Set (sorted by predicted probability)', fontsize=12)
ax2.set_ylabel('Lift', fontsize=12)
ax2.set_title('Lift Chart', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 100])

# Add annotation for top 10%
top_10_idx = int(len(y_sorted) * 0.1)
top_10_lift = lift[top_10_idx]
ax2.axvline(10, color='red', linestyle=':', alpha=0.5)
ax2.text(12, top_10_lift, f'Lift at 10%: {top_10_lift:.2f}x',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/lift_and_gains_charts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lift_and_gains_charts.png")

# Print lift statistics
print(f"\n{'=' * 60}")
print("LIFT CHART STATISTICS")
print(f"{'=' * 60}")
print(f"Lift at 10% of test set: {lift[int(len(y_sorted) * 0.1)]:.2f}x")
print(f"Lift at 20% of test set: {lift[int(len(y_sorted) * 0.2)]:.2f}x")
print(f"Lift at 30% of test set: {lift[int(len(y_sorted) * 0.3)]:.2f}x")
print(f"\nGains at 10%: {gains[int(len(y_sorted) * 0.1)]:.1f}% of attrition cases captured")
print(f"Gains at 20%: {gains[int(len(y_sorted) * 0.2)]:.1f}% of attrition cases captured")
print(f"Gains at 30%: {gains[int(len(y_sorted) * 0.3)]:.1f}% of attrition cases captured")

# ------------------------------
# VISUALIZATION 3: Learning Curves
# ------------------------------
print(f"\n{'=' * 60}")
print("GENERATING LEARNING CURVES")
print(f"{'=' * 60}")

# Create a simpler pipeline for learning curves (without SMOTE to see true performance)
learning_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        C=study.best_params["C"],
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ))
])

train_sizes = np.linspace(0.1, 1.0, 10)

# Calculate learning curves
train_sizes_abs, train_scores, val_scores = learning_curve(
    learning_pipeline,
    X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    shuffle=True
)

# Calculate means and stds
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot learning curves
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

ax.fill_between(train_sizes_abs,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1, color='blue')
ax.fill_between(train_sizes_abs,
                val_scores_mean - val_scores_std,
                val_scores_mean + val_scores_std,
                alpha=0.1, color='red')

ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue',
        linewidth=2, label='Training F1 Score')
ax.plot(train_sizes_abs, val_scores_mean, 'o-', color='red',
        linewidth=2, label='Cross-Validation F1 Score')

ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Learning Curves (F1 Score)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

# Add interpretation text
gap = train_scores_mean[-1] - val_scores_mean[-1]
if gap > 0.1:
    diagnosis = "High variance (overfitting)\nConsider: more data, regularization"
elif val_scores_mean[-1] < 0.6:
    diagnosis = "High bias (underfitting)\nConsider: more features, complex model"
else:
    diagnosis = "Good fit"

ax.text(0.02, 0.98, f"Train-Val Gap: {gap:.3f}\n{diagnosis}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: learning_curves.png")

print(f"\nFinal training score: {train_scores_mean[-1]:.4f} ± {train_scores_std[-1]:.4f}")
print(f"Final validation score: {val_scores_mean[-1]:.4f} ± {val_scores_std[-1]:.4f}")
print(f"Train-validation gap: {gap:.4f}")

# ------------------------------
# VISUALIZATION 4: Confusion Matrices Comparison
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Default threshold
cm_default = confusion_matrix(y_test, y_pred_default)
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_title(f'Default Threshold (0.5)\nF1: {metrics_default["f1"]:.3f} | Kappa: {metrics_default["kappa"]:.3f}',
              fontsize=12, fontweight='bold')

# Optimal threshold
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
ax2.set_ylabel('Actual', fontsize=12)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_title(
    f'Optimal Threshold ({optimal_threshold:.3f})\nF1: {metrics_optimal["f1"]:.3f} | Kappa: {metrics_optimal["kappa"]:.3f}',
    fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices_comparison.png")

# ------------------------------
# Summary Report
# ------------------------------
print(f"\n{'=' * 60}")
print("FINAL SUMMARY")
print(f"{'=' * 60}")
print(f"\nImprovement from Default to Optimal Threshold:")
print(
    f"  Recall (Sensitivity): {metrics_default['recall']:.4f} → {metrics_optimal['recall']:.4f} ({(metrics_optimal['recall'] - metrics_default['recall']) * 100:+.2f}%)")
print(
    f"  Precision          : {metrics_default['precision']:.4f} → {metrics_optimal['precision']:.4f} ({(metrics_optimal['precision'] - metrics_default['precision']) * 100:+.2f}%)")
print(
    f"  F1-Score           : {metrics_default['f1']:.4f} → {metrics_optimal['f1']:.4f} ({(metrics_optimal['f1'] - metrics_default['f1']) * 100:+.2f}%)")
print(
    f"  Cohen's Kappa      : {metrics_default['kappa']:.4f} → {metrics_optimal['kappa']:.4f} ({(metrics_optimal['kappa'] - metrics_default['kappa']) * 100:+.2f}%)")

print(f"\n✓ All visualizations saved to /mnt/user-data/outputs/")
print(f"  - roc_curve_youdens_j.png")
print(f"  - lift_and_gains_charts.png")
print(f"  - learning_curves.png")
print(f"  - confusion_matrices_comparison.png")

plt.close('all')
print("\nAnalysis complete!")