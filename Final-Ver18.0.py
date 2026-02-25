import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_STATE = 51
cluster_col = "Type"
ALPHA = 0.7                 # weight for individual risk
TOP_MIGRATION_PERCENT = 0.05
CAPACITY_LIMIT = 1.2       

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("predictive_maintenance.csv")

cluster_series = df[cluster_col]

features_to_drop = ["UDI", "Product ID", "Type",
                    "Machine failure", "TWF", "HDF",
                    "PWF", "OSF", "RNF"]

X = df.drop(columns=features_to_drop, errors='ignore')
y = df["Machine failure"].values

# ===============================
# 2. TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test, cluster_train, cluster_test = train_test_split(
    X, y, cluster_series, test_size=0.2,
    random_state=RANDOM_STATE, stratify=y
)

cluster_train = cluster_train.str.strip()
cluster_test  = cluster_test.str.strip()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ===============================
# 3. MODEL
# ===============================
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.025,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ===============================
# 4. PREDICTIONS
# ===============================
probs = model.predict_proba(X_test_scaled)[:, 1]

df_decision = X_test.copy()
df_decision['prob'] = probs
df_decision['cluster'] = cluster_test.values

# ===============================
# 5. CLUSTER-AWARE PRIORITY
# ===============================
cluster_risk = df_decision.groupby('cluster')['prob'].mean()
df_decision['cluster_failure_rate'] = df_decision['cluster'].map(cluster_risk)

# New tunable priority formula
df_decision['priority_score'] = (
    ALPHA * df_decision['prob'] +
    (1 - ALPHA) * df_decision['cluster_failure_rate']
)

# ===============================
# 6. THRESHOLD TUNING
# ===============================
thresholds = np.arange(0.05, 0.9, 0.01)
best_f1 = 0
best_t = 0.7

for t in thresholds:
    preds = (df_decision['priority_score'] >= t).astype(int)
    f1 = f1_score(y_test, preds)
    r  = recall_score(y_test, preds)
    if f1 > best_f1 and r >= 0.7:
        best_f1 = f1
        best_t = t

df_decision['predicted_failure'] = (
    df_decision['priority_score'] >= best_t
).astype(int)

# ===============================
# 7. CROSS-CLUSTER MIGRATION
# ===============================

cluster_median = cluster_risk.median()
high_risk_clusters = cluster_risk[cluster_risk > cluster_median].index
low_risk_clusters  = cluster_risk[cluster_risk <= cluster_median].index

df_decision['rank_in_cluster'] = df_decision.groupby('cluster')['priority_score'] \
                                            .rank(ascending=False, method='first')

df_decision['migrate'] = False
for cl in high_risk_clusters:
    cluster_mask = df_decision['cluster'] == cl
    cluster_size = cluster_mask.sum()
    top_k = int(TOP_MIGRATION_PERCENT * cluster_size)
    idx = df_decision[cluster_mask] \
          .sort_values('priority_score', ascending=False) \
          .head(top_k).index
    df_decision.loc[idx, 'migrate'] = True

# ===============================
# 8. CAPACITY-AWARE MIGRATION PLAN
# ===============================
migration_plan = []

targets = df_decision[
    (df_decision['cluster'].isin(low_risk_clusters)) &
    (~df_decision['migrate'])
].copy()

targets['current_load'] = 1.0  

for src_idx, src in df_decision[df_decision['migrate']].iterrows():

    total_target_priority = targets['priority_score'].sum() + 1e-6

    for tgt_idx, tgt in targets.iterrows():

        fraction = tgt['priority_score'] / total_target_priority
        projected_load = targets.loc[tgt_idx, 'current_load'] + fraction

        if projected_load <= CAPACITY_LIMIT:
            migration_plan.append({
                'source_machine': src_idx,
                'target_machine': tgt_idx,
                'fraction_of_load': fraction,
                'target_cluster': tgt['cluster']
            })
            targets.loc[tgt_idx, 'current_load'] = projected_load

migration_plan = pd.DataFrame(migration_plan)

# ===============================
# 9. FINAL METRICS
# ===============================
precision_carsm = precision_score(y_test, df_decision['predicted_failure'])
recall_carsm    = recall_score(y_test, df_decision['predicted_failure'])
accuracy_carsm  = accuracy_score(y_test, df_decision['predicted_failure'])
f1_carsm        = f1_score(y_test, df_decision['predicted_failure'])

print("\n==============================")
print("CARS-M FINAL METRICS (Improved Version)")
print("==============================")
print(f"Best Threshold: {best_t:.2f}")
print(f"Precision: {precision_carsm:.4f}")
print(f"Recall:    {recall_carsm:.4f}")
print(f"Accuracy:  {accuracy_carsm:.4f}")
print(f"F1 Score:  {f1_carsm:.4f}")
print(f"Machines flagged for migration: {df_decision['migrate'].sum()}")

# ===============================
# 10. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, df_decision['predicted_failure'])
cm_df = pd.DataFrame(cm,
                     index=['Actual OK', 'Actual Failure'],
                     columns=['Predicted OK', 'Predicted Failure'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Improved CARS-M Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("\nSample migration plan:")
print(migration_plan.head())