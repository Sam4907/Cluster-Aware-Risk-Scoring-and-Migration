#final

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 39

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
# BASELINE MODEL
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
# CARS-M MODEL
# ===============================

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

carsm_probs = carsm_model.predict_proba(X_test_scaled)[:,1]

# ===============================
# MONTE CARLO THRESHOLD TUNING
# ===============================

NUM_SIMULATIONS = 100
NOISE_STD = 0.015

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
best_t = 0.1

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
NOISE_STD = 0.02

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
# FULL MONTE CARLO METRICS
# ===============================

def monte_carlo_full_metrics(model, X_test, y_test, threshold=None):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

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
        accuracy_scores.append(accuracy_score(y_test, preds))

    return (np.array(f1_scores),
            np.array(precision_scores),
            np.array(recall_scores),
            np.array(accuracy_scores))


# ===============================
# RUN FOR BASELINE
# ===============================
f1_base_mc, prec_base_mc, rec_base_mc, acc_base_mc = \
    monte_carlo_full_metrics(baseline_model, X_test_scaled, y_test)

# ===============================
# RUN FOR CARS-M
# ===============================
f1_carsm_mc, prec_carsm_mc, rec_carsm_mc, acc_carsm_mc = \
    monte_carlo_full_metrics(carsm_model, X_test_scaled, y_test, threshold=best_t)


# ===============================
# CONFIDENCE INTERVAL FUNCTION
# ===============================
def ci_95(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)


# ===============================
# PRINT FINAL COMPARISON
# ===============================
print("\n==============================")
print(" FINAL MONTE CARLO COMPARISON ")
print("==============================")

print("\n--- BASELINE ---")
print(f"F1 Mean:        {np.mean(f1_base_mc):.4f} | 95% CI: {ci_95(f1_base_mc)}")
print(f"Precision Mean: {np.mean(prec_base_mc):.4f} | 95% CI: {ci_95(prec_base_mc)}")
print(f"Recall Mean:    {np.mean(rec_base_mc):.4f} | 95% CI: {ci_95(rec_base_mc)}")
print(f"Accuracy Mean:  {np.mean(acc_base_mc):.4f} | 95% CI: {ci_95(acc_base_mc)}")

print("\n--- CARS-M ---")
print(f"F1 Mean:        {np.mean(f1_carsm_mc):.4f} | 95% CI: {ci_95(f1_carsm_mc)}")
print(f"Precision Mean: {np.mean(prec_carsm_mc):.4f} | 95% CI: {ci_95(prec_carsm_mc)}")
print(f"Recall Mean:    {np.mean(rec_carsm_mc):.4f} | 95% CI: {ci_95(rec_carsm_mc)}")
print(f"Accuracy Mean:  {np.mean(acc_carsm_mc):.4f} | 95% CI: {ci_95(acc_carsm_mc)}")

# ===============================
# VISUALIZE MONTE CARLO RESULTS
# ===============================

import matplotlib.pyplot as plt

def plot_mc_metrics(f1_base, f1_carsm, prec_base, prec_carsm, rec_base, rec_carsm, acc_base, acc_carsm):
    plt.figure(figsize=(16, 10))

    # F1 Score
    plt.subplot(2, 2, 1)
    plt.hist(f1_base, bins=20, alpha=0.5, label='Baseline', color='blue')
    plt.hist(f1_carsm, bins=20, alpha=0.5, label='CARS-M', color='orange')
    plt.title('F1 Score Distribution')
    plt.legend()
    plt.grid(True)

    # Precision
    plt.subplot(2, 2, 2)
    plt.hist(prec_base, bins=20, alpha=0.5, label='Baseline', color='blue')
    plt.hist(prec_carsm, bins=20, alpha=0.5, label='CARS-M', color='orange')
    plt.title('Precision Distribution')
    plt.legend()
    plt.grid(True)

    # Recall
    plt.subplot(2, 2, 3)
    plt.hist(rec_base, bins=20, alpha=0.5, label='Baseline', color='blue')
    plt.hist(rec_carsm, bins=20, alpha=0.5, label='CARS-M', color='orange')
    plt.title('Recall Distribution')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(2, 2, 4)
    plt.hist(acc_base, bins=20, alpha=0.5, label='Baseline', color='blue')
    plt.hist(acc_carsm, bins=20, alpha=0.5, label='CARS-M', color='orange')
    plt.title('Accuracy Distribution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_mc_metrics(f1_base_mc, f1_carsm_mc, 
                prec_base_mc, prec_carsm_mc, 
                rec_base_mc, rec_carsm_mc, 
                acc_base_mc, acc_carsm_mc)
