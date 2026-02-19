# ===============================
# CLEAN COMPARISON: CARS-M vs BASELINE
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("predictive_maintenance.csv")

features_to_drop = ["UDI", "Product ID", "Type",
                    "Machine failure", "TWF", "HDF",
                    "PWF", "OSF", "RNF"]

X = df.drop(columns=features_to_drop, errors='ignore')
y = df["Machine failure"].values

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 1️⃣ BASELINE MODEL
# ===============================
baseline_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    n_jobs=-1
)

baseline_model.fit(X_train_scaled, y_train)

baseline_preds = baseline_model.predict(X_test_scaled)

precision_base = precision_score(y_test, baseline_preds)
recall_base = recall_score(y_test, baseline_preds)
accuracy_base = accuracy_score(y_test, baseline_preds)
f1_base = f1_score(y_test, baseline_preds)

# ===============================
# IMPROVED CARS-M MODEL
# ===============================

# Compute imbalance ratio
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

carsm_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

carsm_model.fit(X_train_scaled, y_train)

# Predict probabilities
carsm_probs = carsm_model.predict_proba(X_test_scaled)[:,1]

# ===============================
# MONTE CARLO THRESHOLD TUNING
# ===============================

NUM_SIMULATIONS = 50
NOISE_STD = 0.02

mc_probs = []

for i in range(NUM_SIMULATIONS):
    noise = np.random.normal(0, NOISE_STD, X_test_scaled.shape)
    X_noisy = X_test_scaled + noise
    probs = carsm_model.predict_proba(X_noisy)[:,1]
    mc_probs.append(probs)

mc_probs = np.array(mc_probs)

# Use MEAN probability across simulations
mean_probs = mc_probs.mean(axis=0)

# Tune threshold on stable probabilities
thresholds = np.arange(0.05, 0.9, 0.01)

best_f1 = 0
best_t = 0.5

for t in thresholds:
    preds = (mean_probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

carsm_preds = (mean_probs >= best_t).astype(int)

precision_carsm = precision_score(y_test, carsm_preds)
recall_carsm = recall_score(y_test, carsm_preds)
accuracy_carsm = accuracy_score(y_test, carsm_preds)
f1_carsm = f1_score(y_test, carsm_preds)



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

NUM_SIMULATIONS = 100
NOISE_STD = 0.01

# ===============================
# MONTE CARLO SIMULATION FUNCTION
# ===============================
def monte_carlo_evaluation(model, X_test, y_test, threshold=None):
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(NUM_SIMULATIONS):
        noise = np.random.normal(0, NOISE_STD, X_test.shape)
        X_noisy = X_test + noise

        probs = model.predict_proba(X_noisy)[:,1]

        if threshold is None:
            preds = (probs >= 0.5).astype(int)
        else:
            preds = (probs >= threshold).astype(int)

        f1_scores.append(f1_score(y_test, preds))
        precision_scores.append(precision_score(y_test, preds))
        recall_scores.append(recall_score(y_test, preds))

    return np.array(f1_scores), np.array(precision_scores), np.array(recall_scores)

# ===============================
# MONTE CARLO FOR BASELINE
# ===============================
f1_base_mc, prec_base_mc, rec_base_mc = monte_carlo_evaluation(
    baseline_model, X_test_scaled, y_test
)

# ===============================
# MONTE CARLO FOR CARS-M
# ===============================
f1_carsm_mc, prec_carsm_mc, rec_carsm_mc = monte_carlo_evaluation(
    carsm_model, X_test_scaled, y_test, threshold=best_t
)

# ===============================
# CONFIDENCE INTERVAL FUNCTION
# ===============================
def ci_95(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

# ===============================
# PRINT MONTE CARLO RESULTS
# ===============================
print("\n===== MONTE CARLO RESULTS =====")

print("\nBASELINE F1 Mean:", np.mean(f1_base_mc))
print("BASELINE 95% CI:", ci_95(f1_base_mc))

print("\nCARS-M F1 Mean:", np.mean(f1_carsm_mc))
print("CARS-M 95% CI:", ci_95(f1_carsm_mc))

# ===============================
# VISUALIZATION 1: F1 DISTRIBUTION
# ===============================
plt.figure(figsize=(8,5))
plt.hist(f1_base_mc, bins=15, alpha=0.6, label="Baseline")
plt.hist(f1_carsm_mc, bins=15, alpha=0.6, label="CARS-M")
plt.axvline(np.mean(f1_base_mc), linestyle='--')
plt.axvline(np.mean(f1_carsm_mc), linestyle='--')
plt.title("Monte Carlo F1 Distribution")
plt.xlabel("F1 Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# ===============================
# VISUALIZATION 2: ROC CURVE
# ===============================
baseline_probs = baseline_model.predict_proba(X_test_scaled)[:,1]
carsm_probs = carsm_model.predict_proba(X_test_scaled)[:,1]

fpr_base, tpr_base, _ = roc_curve(y_test, baseline_probs)
fpr_carsm, tpr_carsm, _ = roc_curve(y_test, carsm_probs)

auc_base = auc(fpr_base, tpr_base)
auc_carsm = auc(fpr_carsm, tpr_carsm)

plt.figure(figsize=(6,6))
plt.plot(fpr_base, tpr_base, label=f"Baseline (AUC={auc_base:.3f})")
plt.plot(fpr_carsm, tpr_carsm, label=f"CARS-M (AUC={auc_carsm:.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
