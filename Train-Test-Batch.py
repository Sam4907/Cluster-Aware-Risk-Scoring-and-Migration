# ===============================
# HYBRID CLOUD-EDGE SERVER RELIABILITY
# CARS-M + MONTE CARLO + MIGRATION + INSIGHTFUL VISUALIZATIONS
# WITH TRAIN/TEST SPLIT
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
BASE_RISK_THRESHOLD = 0.45
NUM_SIMULATIONS = 10
RANDOM_STATE = 39
SCALE_POS_WEIGHT = 4

# Priority settings
RECALL_PRIORITY_FACTOR = -0.02  # slightly penalize recall to reduce false positives
MIN_RECALL_TARGET = 0.80        # minimum recall to maintain

# Migration parameters
MIGRATION_RISK_THRESHOLD = 0.8
MIGRATION_SAFETY_FACTOR = 0.9

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("predictive_maintenance.csv")
features_to_drop = ["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

X_raw = df.drop(columns=features_to_drop, errors='ignore')
y_raw = df["Machine failure"].values

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_raw
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===============================
# FEATURE ENGINEERING
# ===============================
def compute_features_np(X):
    anomaly = np.abs(X - X.mean(axis=0))
    degradation = np.gradient(X, axis=0)
    stress = X
    rolling_mean = np.empty_like(X)
    rolling_mean[0] = X[0]
    rolling_mean[1] = X[:2].mean(axis=0)
    for i in range(2, X.shape[0]):
        rolling_mean[i] = X[i-2:i+1].mean(axis=0)
    return np.hstack([anomaly, degradation, stress, rolling_mean])

X_train_features = compute_features_np(X_train_scaled)
X_test_features  = compute_features_np(X_test_scaled)

# ===============================
# TRAIN MODEL
# ===============================
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    scale_pos_weight=SCALE_POS_WEIGHT,
    n_jobs=-1
)
xgb_model.fit(X_train_features, y_train)

# ===============================
# THRESHOLD TUNING
# ===============================
def threshold_tuning_f1(probs, y_true, thresholds=np.arange(0.3, 0.9, 0.01), min_recall=0.6):
    best_threshold = thresholds[0]
    best_f1 = -1
    for t in thresholds:
        preds = (probs >= t).astype(int)
        recall = recall_score(y_true, preds)
        if recall < min_recall:
            continue
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold

# Compute threshold on training set
probs_train = xgb_model.predict_proba(X_train_features)[:,1]
best_threshold = threshold_tuning_f1(probs_train, y_train)

# ===============================
# MIGRATION FUNCTION
# ===============================
def migrate_clusters(R_cluster, risk_threshold=MIGRATION_RISK_THRESHOLD):
    num_clusters = R_cluster.shape[0]
    migration_plan = np.arange(num_clusters)
    high_risk = R_cluster > risk_threshold
    if high_risk.any():
        safest_cluster = np.argmin(R_cluster)
        migration_plan[high_risk] = safest_cluster
    return migration_plan

# ===============================
# MONTE CARLO SIMULATION FUNCTION
# ===============================
def run_monte_carlo(X_features, y_true):
    num_samples, num_features = X_features.shape
    noise = np.random.normal(0, 0.05, (NUM_SIMULATIONS, num_samples, num_features))
    X_features_mc_flat = (X_features[None, :, :] + noise).reshape(NUM_SIMULATIONS*num_samples, -1)

    # Predict probabilities
    probs_flat = xgb_model.predict_proba(X_features_mc_flat)[:,1]
    probs_mc = probs_flat.reshape(NUM_SIMULATIONS, num_samples)

    # Apply priority adjustment
    R_cluster_mc = probs_mc * (1 + RECALL_PRIORITY_FACTOR)

    # Apply migration
    migration_plans = np.empty_like(R_cluster_mc, dtype=int)
    for i in range(NUM_SIMULATIONS):
        migration_plans[i] = migrate_clusters(R_cluster_mc[i])
        migrated = migration_plans[i] != np.arange(num_samples)
        R_cluster_mc[i, migrated] *= MIGRATION_SAFETY_FACTOR

    # Predictions
    preds_mc = (R_cluster_mc >= best_threshold).astype(int)

    # Metrics
    precision_results = [precision_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]
    recall_results    = [recall_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]
    accuracy_results  = [accuracy_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]
    f1_results        = [f1_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]

    return precision_results, recall_results, accuracy_results, f1_results, R_cluster_mc, migration_plans

# ===============================
# RUN SIMULATIONS
# ===============================
precision_train, recall_train, accuracy_train, f1_train, R_train_mc, migration_train = run_monte_carlo(X_train_features, y_train)
precision_test, recall_test, accuracy_test, f1_test, R_test_mc, migration_test = run_monte_carlo(X_test_features, y_test)

# Confidence interval
def ci_95(data):
    mean_val = np.mean(data)
    ci_lower = mean_val - 1.96*np.std(data)/np.sqrt(len(data))
    ci_upper = mean_val + 1.96*np.std(data)/np.sqrt(len(data))
    return ci_lower, ci_upper

# ===============================
# PRINT METRICS
# ===============================
print("\n===== TRAIN SET =====")
print(f"Precision: {np.mean(precision_train):.4f}, 95% CI: {ci_95(precision_train)}")
print(f"Recall:    {np.mean(recall_train):.4f}, 95% CI: {ci_95(recall_train)}")
print(f"Accuracy:  {np.mean(accuracy_train):.4f}, 95% CI: {ci_95(accuracy_train)}")
print(f"F1 Score:  {np.mean(f1_train):.4f}, 95% CI: {ci_95(f1_train)}")

print("\n===== TEST SET =====")
print(f"Precision: {np.mean(precision_test):.4f}, 95% CI: {ci_95(precision_test)}")
print(f"Recall:    {np.mean(recall_test):.4f}, 95% CI: {ci_95(recall_test)}")
print(f"Accuracy:  {np.mean(accuracy_test):.4f}, 95% CI: {ci_95(accuracy_test)}")
print(f"F1 Score:  {np.mean(f1_test):.4f}, 95% CI: {ci_95(f1_test)}")

# ===============================
# OPTIONAL VISUALIZATIONS
# ===============================
# Example: Monte Carlo variability for test set
plt.figure(figsize=(10,5))
plt.bar(range(len(R_test_mc[0])), R_test_mc.std(axis=0), alpha=0.7, color='purple')
plt.xlabel("Cluster Index")
plt.ylabel("Risk Std Dev")
plt.title("Monte Carlo Risk Variability per Cluster (Test Set)")
plt.grid(True)
plt.show()
