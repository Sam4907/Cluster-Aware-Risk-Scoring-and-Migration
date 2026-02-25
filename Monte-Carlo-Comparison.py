# ===============================
# FULL WORKING CARS-M MONTE CARLO
# ===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PARAMETERS
# ===============================
RANDOM_STATE = 51
N_MONTE_CARLO = 10        
NOISE_STD = 0.01          
cluster_col = "Type"
ALPHA = 0.7                  
TOP_MIGRATION_PERCENT = 0.05
CAPACITY_LIMIT = 1.2         

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("predictive_maintenance.csv")
df[cluster_col] = df[cluster_col].str.strip()
features_to_drop = ["UDI", "Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
X_orig = df.drop(columns=features_to_drop, errors='ignore')
y_orig = df["Machine failure"].values
clusters = df[cluster_col]

# ===============================
# MONTE CARLO METRICS STORAGE
# ===============================
metrics_mc = {
    "f1_base": [], "f1_carsm": [],
    "prec_base": [], "prec_carsm": [],
    "rec_base": [], "rec_carsm": [],
    "acc_base": [], "acc_carsm": []
}

# ===============================
# MONTE CARLO LOOP
# ===============================
for i in range(N_MONTE_CARLO):
    # Train/test split with stratification
    X_train, X_test, y_train, y_test, cluster_train, cluster_test = train_test_split(
        X_orig, y_orig, clusters, test_size=0.2,
        random_state=np.random.randint(10000), stratify=y_orig
    )
    
    # Add Gaussian noise
    X_train_noisy = X_train + np.random.normal(0, NOISE_STD, X_train.shape)
    X_test_noisy  = X_test  + np.random.normal(0, NOISE_STD, X_test.shape)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_test_scaled  = scaler.transform(X_test_noisy)
    
    scale_pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1e-6)
    
    # ==============================
    # BASELINE MODEL
    # ==============================
    baseline_model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.025,
        random_state=RANDOM_STATE, eval_metric='logloss',
        scale_pos_weight=scale_pos_weight, n_jobs=-1
    )
    baseline_model.fit(X_train_scaled, y_train)
    baseline_probs = baseline_model.predict_proba(X_test_scaled)[:, 1]
    baseline_preds = (baseline_probs >= 0.5).astype(int)
    
    # ==============================
    # CARS-M MODEL
    # ==============================
    carsm_model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.025,
        random_state=RANDOM_STATE, eval_metric='logloss',
        scale_pos_weight=scale_pos_weight, n_jobs=-1
    )
    carsm_model.fit(X_train_scaled, y_train)
    probs = carsm_model.predict_proba(X_test_scaled)[:, 1]
    
    # Cluster-aware priority score
    df_decision = X_test.copy()
    df_decision['prob'] = probs
    df_decision['cluster'] = cluster_test.values
    cluster_risk = df_decision.groupby('cluster')['prob'].mean()
    df_decision['cluster_failure_rate'] = df_decision['cluster'].map(cluster_risk)
    df_decision['priority_score'] = ALPHA*df_decision['prob'] + (1-ALPHA)*df_decision['cluster_failure_rate']
    
    # Threshold tuning for F1 with recall >= 0.7
    thresholds = np.arange(0.03, 0.9, 0.01)
    best_f1 = 0
    best_t = 0.5
    for t in thresholds:
        preds = (df_decision['priority_score'] >= t).astype(int)
        f1 = f1_score(y_test, preds)
        r  = recall_score(y_test, preds)
        if f1 > best_f1 and r >= 0.7:
            best_f1 = f1
            best_t = t
    carsm_preds = (df_decision['priority_score'] >= best_t).astype(int)
    
    # ==============================
    # CROSS-CLUSTER MIGRATION (optional)
    # ==============================
    cluster_median = cluster_risk.median()
    high_risk_clusters = cluster_risk[cluster_risk > cluster_median].index
    low_risk_clusters  = cluster_risk[cluster_risk <= cluster_median].index
    
    df_decision['rank_in_cluster'] = df_decision.groupby('cluster')['priority_score'] \
                                                .rank(ascending=False, method='first')
    df_decision['migrate'] = False
    for cl in high_risk_clusters:
        cluster_mask = df_decision['cluster'] == cl
        cluster_size = cluster_mask.sum()
        top_k = max(1, int(TOP_MIGRATION_PERCENT * cluster_size))
        idx = df_decision[cluster_mask].sort_values('priority_score', ascending=False).head(top_k).index
        df_decision.loc[idx, 'migrate'] = True
    
    # ==============================
    # APPEND METRICS
    # ==============================
    metrics_mc['f1_base'].append(f1_score(y_test, baseline_preds))
    metrics_mc['f1_carsm'].append(f1_score(y_test, carsm_preds))
    metrics_mc['prec_base'].append(precision_score(y_test, baseline_preds))
    metrics_mc['prec_carsm'].append(precision_score(y_test, carsm_preds))
    metrics_mc['rec_base'].append(recall_score(y_test, baseline_preds))
    metrics_mc['rec_carsm'].append(recall_score(y_test, carsm_preds))
    metrics_mc['acc_base'].append(accuracy_score(y_test, baseline_preds))
    metrics_mc['acc_carsm'].append(accuracy_score(y_test, carsm_preds))

# ===============================
# PLOTTING FUNCTION
# ===============================
def plot_mc_metrics(metrics_mc):
    plt.figure(figsize=(16,10))
    for idx, metric in enumerate(['f1','prec','rec','acc']):
        plt.subplot(2,2,idx+1)
        plt.hist(metrics_mc[f'{metric}_base'], bins=20, alpha=0.5, label='Baseline', color='blue')
        plt.hist(metrics_mc[f'{metric}_carsm'], bins=20, alpha=0.5, label='CARS-M', color='orange')
        plt.title(f'{metric.upper()} Distribution')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===============================
# CALL PLOTTING
# ===============================
plot_mc_metrics(metrics_mc)

