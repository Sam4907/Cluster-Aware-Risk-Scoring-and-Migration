# ===============================
# HYBRID CLOUD-EDGE SERVER RELIABILITY
# CARS-M + MONTE CARLO + MIGRATION + INSIGHTFUL VISUALIZATIONS
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ===============================
# PARAMETERS
# ===============================
BASE_RISK_THRESHOLD = 0.45
DELTA = 0.05
NUM_SIMULATIONS = 10
RANDOM_STATE = 42
SCALE_POS_WEIGHT = 4

# Priority settings
RECALL_PRIORITY_FACTOR = -0.05  # slightly penalize recall to reduce false positives
MIN_RECALL_TARGET = 0.80        # minimum recall to maintain

# Migration parameters
MIGRATION_RISK_THRESHOLD = 0.8
MIGRATION_SAFETY_FACTOR = 0.9

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("predictive_maintenance.csv")
y_true = df["Machine failure"].values
features_to_drop = ["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
X = df.drop(columns=features_to_drop, errors='ignore')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

X_features = compute_features_np(X_scaled)

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
xgb_model.fit(X_features, y_true)

# ===============================
# THRESHOLD TUNING
# ===============================
def threshold_tuning_f1(probs, y_true, thresholds=np.arange(0.3, 0.9, 0.01), min_recall=0.88):
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

probs_raw = xgb_model.predict_proba(X_features)[:,1]
best_threshold = threshold_tuning_f1(probs_raw, y_true)

# ===============================
# VECTORISED MONTE CARLO SIMULATION
# ===============================
num_samples, num_features = X_scaled.shape
noise = np.random.normal(0, 0.05, (NUM_SIMULATIONS, num_samples, num_features))
X_noisy = X_scaled[None, :, :] + noise

anomaly = np.abs(X_noisy - X_scaled.mean(axis=0))
degradation = np.gradient(X_noisy, axis=1)
stress = X_noisy

rolling_mean = np.empty_like(X_noisy)
rolling_mean[:,0,:] = X_noisy[:,0,:]
rolling_mean[:,1,:] = X_noisy[:,0:2,:].mean(axis=1)
for i in range(2, num_samples):
    rolling_mean[:,i,:] = X_noisy[:,i-2:i+1,:].mean(axis=1)

X_features_mc = np.concatenate([anomaly, degradation, stress, rolling_mean], axis=2)
X_features_mc_flat = X_features_mc.reshape(NUM_SIMULATIONS*num_samples, -1)

probs_flat = xgb_model.predict_proba(X_features_mc_flat)[:,1]
probs_mc = probs_flat.reshape(NUM_SIMULATIONS, num_samples)
R_cluster_mc = probs_mc * (1 + RECALL_PRIORITY_FACTOR)

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

migration_plans = np.empty_like(R_cluster_mc, dtype=int)
for i in range(NUM_SIMULATIONS):
    migration_plans[i] = migrate_clusters(R_cluster_mc[i], risk_threshold=MIGRATION_RISK_THRESHOLD)
    migrated = migration_plans[i] != np.arange(num_samples)
    R_cluster_mc[i, migrated] *= MIGRATION_SAFETY_FACTOR

# ===============================
# METRICS
# ===============================
preds_mc = (R_cluster_mc >= best_threshold).astype(int)

precision_results = [precision_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]
recall_results = [recall_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]
accuracy_results = [accuracy_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]
f1_results = [f1_score(y_true, preds_mc[i]) for i in range(NUM_SIMULATIONS)]

def ci_95(data):
    mean_val = np.mean(data)
    ci_lower = mean_val - 1.96*np.std(data)/np.sqrt(len(data))
    ci_upper = mean_val + 1.96*np.std(data)/np.sqrt(len(data))
    return ci_lower, ci_upper

print("\n===== PRECISION-FIRST VECTORISED MONTE CARLO WITH MIGRATION =====")
print(f"Precision: {np.mean(precision_results):.4f}, 95% CI: {ci_95(precision_results)}")
print(f"Recall:    {np.mean(recall_results):.4f}, 95% CI: {ci_95(recall_results)}")
print(f"Accuracy:  {np.mean(accuracy_results):.4f}, 95% CI: {ci_95(accuracy_results)}")
print(f"F1 Score:  {np.mean(f1_results):.4f}, 95% CI: {ci_95(f1_results)}")

# ===============================
# VISUALIZATIONS
# ===============================

# 1. Metrics histograms
plt.figure(figsize=(12,6))
plt.subplot(1,4,1); plt.hist(precision_results, bins=20, color='blue', alpha=0.7); plt.title('Precision')
plt.subplot(1,4,2); plt.hist(recall_results, bins=20, color='green', alpha=0.7); plt.title('Recall')
plt.subplot(1,4,3); plt.hist(accuracy_results, bins=20, color='red', alpha=0.7); plt.title('Accuracy')
plt.subplot(1,4,4); plt.hist(f1_results, bins=20, color='purple', alpha=0.7); plt.title('F1 Score')
plt.tight_layout()
plt.show()

# 2. Migration count per simulation
migration_counts = (migration_plans != np.arange(num_samples)).sum(axis=1)
plt.figure(figsize=(8,5))
plt.hist(migration_counts, bins=20, color='orange', alpha=0.7)
plt.title("Number of Clusters Migrated per Simulation")
plt.xlabel("Clusters Migrated")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 3. Risk distribution before vs after migration
avg_risk_before = R_cluster_mc / MIGRATION_SAFETY_FACTOR
avg_risk_after = R_cluster_mc

plt.figure(figsize=(8,5))
plt.hist(avg_risk_before.flatten(), bins=50, alpha=0.5, label='Before Migration', density=True, color='red')
plt.hist(avg_risk_after.flatten(), bins=50, alpha=0.5, label='After Migration', density=True, color='green')
plt.title("Cluster Risk Distribution Before vs After Migration")
plt.xlabel("Cluster Risk")
plt.ylabel("Density")
plt.legend()
plt.show()

# 4. Monte Carlo risk variability per cluster
risk_std_per_cluster = R_cluster_mc.std(axis=0)
plt.figure(figsize=(10,5))
plt.bar(range(num_samples), risk_std_per_cluster, color='purple', alpha=0.7)
plt.xlabel("Cluster Index")
plt.ylabel("Risk Standard Deviation")
plt.title("Monte Carlo Risk Variability per Cluster")
plt.grid(True)
plt.show()

# 5. Precision-Recall scatter across simulations
plt.figure(figsize=(8,6))
plt.scatter(recall_results, precision_results, alpha=0.6, color='blue')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall Across Monte Carlo Simulations")
plt.grid(True)
plt.show()

# 6. Cluster risk before vs after migration (line plot)
plt.figure(figsize=(12,6))
plt.plot(avg_risk_before.mean(axis=0), label='Before Migration', marker='o', linestyle='--')
plt.plot(avg_risk_after.mean(axis=0), label='After Migration', marker='o', linestyle='-')
plt.xlabel('Cluster Index')
plt.ylabel('Average Risk')
plt.title('Cluster Risk Before vs After Migration')
plt.legend()
plt.grid(True)
plt.show()
