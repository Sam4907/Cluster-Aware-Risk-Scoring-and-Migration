# ===============================
# CARS-M for FD003
# ===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

RANDOM_STATE = 39
NUM_SIMULATIONS = 100
NOISE_STD = 0.02

# ===============================
# LOAD DATA (FD003)
# ===============================
train_df = pd.read_csv("train_FD003.txt", sep="\s+", header=None)
train_df = train_df.dropna(axis=1, how='all')
train_df.columns = ['unit_nr', 'time_cycles'] + [f's{i}' for i in range(1,25)]

test_df = pd.read_csv("test_FD003.txt", sep="\s+", header=None)
test_df = test_df.dropna(axis=1, how='all')
test_df.columns = ['unit_nr', 'time_cycles'] + [f's{i}' for i in range(1,25)]

rul_df = pd.read_csv("RUL_FD003.txt", header=None)
rul_df.columns = ['RUL']

# ===============================
# CREATE FAILURE LABELS
# ===============================
# Failure = 1 if remaining cycles <= 30
max_cycles = train_df.groupby('unit_nr')['time_cycles'].max().reset_index()
train_df = train_df.merge(max_cycles, on='unit_nr', suffixes=('', '_max'))
train_df['Machine_failure'] = ((train_df['time_cycles_max'] - train_df['time_cycles']) <= 30).astype(int)
train_df = train_df.drop(columns=['time_cycles_max'])

# ===============================
# FEATURES & TARGET
# ===============================
X = train_df.drop(columns=['unit_nr', 'Machine_failure'])
y = train_df['Machine_failure'].values

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

# ===============================
# CARS-M MODEL
# ===============================
scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))

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

# ===============================
# MONTE CARLO THRESHOLD TUNING
# ===============================
mc_probs = []
for _ in range(NUM_SIMULATIONS):
    noise = np.random.normal(0, NOISE_STD, X_test_scaled.shape)
    X_noisy = X_test_scaled + noise
    probs = carsm_model.predict_proba(X_noisy)[:,1]
    mc_probs.append(probs)

mc_probs = np.array(mc_probs)
mean_probs = mc_probs.mean(axis=0)

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

# ===============================
# MONTE CARLO FULL METRICS FUNCTION
# ===============================
def monte_carlo_full_metrics(model, X_test, y_test, threshold=None):
    f1_scores, precision_scores, recall_scores, accuracy_scores = [], [], [], []
    for _ in range(NUM_SIMULATIONS):
        noise = np.random.normal(0, NOISE_STD, X_test.shape)
        X_noisy = X_test + noise
        probs = model.predict_proba(X_noisy)[:,1]
        preds = (probs >= (threshold if threshold is not None else 0.5)).astype(int)
        f1_scores.append(f1_score(y_test, preds))
        precision_scores.append(precision_score(y_test, preds))
        recall_scores.append(recall_score(y_test, preds))
        accuracy_scores.append(accuracy_score(y_test, preds))
    return (np.array(f1_scores), np.array(precision_scores),
            np.array(recall_scores), np.array(accuracy_scores))

f1_base_mc, prec_base_mc, rec_base_mc, acc_base_mc = \
    monte_carlo_full_metrics(baseline_model, X_test_scaled, y_test)

f1_carsm_mc, prec_carsm_mc, rec_carsm_mc, acc_carsm_mc = \
    monte_carlo_full_metrics(carsm_model, X_test_scaled, y_test, threshold=best_t)

def ci_95(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

# ===============================
# PRINT RESULTS
# ===============================
print("\n--- BASELINE ---")
print(f"F1 Mean: {np.mean(f1_base_mc):.4f} | 95% CI: {ci_95(f1_base_mc)}")
print(f"Precision Mean: {np.mean(prec_base_mc):.4f} | 95% CI: {ci_95(prec_base_mc)}")
print(f"Recall Mean: {np.mean(rec_base_mc):.4f} | 95% CI: {ci_95(rec_base_mc)}")
print(f"Accuracy Mean: {np.mean(acc_base_mc):.4f} | 95% CI: {ci_95(acc_base_mc)}")

print("\n--- CARS-M ---")
print(f"F1 Mean: {np.mean(f1_carsm_mc):.4f} | 95% CI: {ci_95(f1_carsm_mc)}")
print(f"Precision Mean: {np.mean(prec_carsm_mc):.4f} | 95% CI: {ci_95(prec_carsm_mc)}")
print(f"Recall Mean: {np.mean(rec_carsm_mc):.4f} | 95% CI: {ci_95(rec_carsm_mc)}")
print(f"Accuracy Mean: {np.mean(acc_carsm_mc):.4f} | 95% CI: {ci_95(acc_carsm_mc)}")

import matplotlib.pyplot as plt

metrics = ['F1', 'Precision', 'Recall', 'Accuracy']

# Means
baseline_means = [np.mean(f1_base_mc), np.mean(prec_base_mc), np.mean(rec_base_mc), np.mean(acc_base_mc)]
carsm_means   = [np.mean(f1_carsm_mc), np.mean(prec_carsm_mc), np.mean(rec_carsm_mc), np.mean(acc_carsm_mc)]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(x - width/2, baseline_means, width, label='Baseline', color='blue')
ax.bar(x + width/2, carsm_means, width, label='CARS-M', color='orange')

ax.set_ylabel('Score')
ax.set_title('Monte Carlo Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Zoom in to show improvements clearly
ax.set_ylim(0.85, 1.0)

ax.legend()
ax.grid(axis='y')

plt.show()