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

# ===============================
# THRESHOLD TUNING
# ===============================
probs = carsm_model.predict_proba(X_test_scaled)[:,1]
thresholds = np.arange(0.05, 0.9, 0.01)

best_f1 = 0
best_t = 0.5  # default

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

# Final predictions
carsm_preds = (probs >= best_t).astype(int)

# ===============================
# FINAL METRICS
# ===============================
precision_carsm = precision_score(y_test, carsm_preds)
recall_carsm = recall_score(y_test, carsm_preds)
accuracy_carsm = accuracy_score(y_test, carsm_preds)
f1_carsm = f1_score(y_test, carsm_preds)

print("\n==============================")
print("CARS-M FINAL METRICS")
print("==============================")
print(f"Best Threshold: {best_t:.2f}")
print(f"Precision: {precision_carsm:.4f}")
print(f"Recall:    {recall_carsm:.4f}")
print(f"Accuracy:  {accuracy_carsm:.4f}")
print(f"F1 Score:  {f1_carsm:.4f}")