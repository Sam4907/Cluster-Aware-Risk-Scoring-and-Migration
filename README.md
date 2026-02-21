# Cluster-Aware Risk Scoring and Migration (CARS-M)

CARS-M is a **risk-aware predictive maintenance framework** designed for hybrid cloud-edge environments. It identifies high-risk system components and enables proactive risk management, minimizing the chance of cascading failures in complex systems.

## Algorithm Overview

CARS-M (Cluster-Aware Risk Scoring and Migration) works as follows:

1. **Cluster-Level Risk Assessment** – System components are grouped into operational clusters, and each cluster’s risk is evaluated.
2. **Failure Probability Estimation** – Uses an XGBoost classifier to predict the likelihood of failure for each component based on historical sensor data.
3. **Threshold Optimization** – Determines optimal decision boundaries to balance recall and precision, ensuring high sensitivity to potential failures while maintaining overall reliability.
4. **Risk Prioritization & Migration** – Components with the highest predicted risk are flagged for preventive action, enabling administrators to allocate resources efficiently and prevent cascading failures.

## Installation

Clone the repository:

```bash
git clone https://github.com/Sam4907/Cluster-Aware-Risk-Scoring-and-Migration.git
cd Cluster-Aware-Risk-Scoring-and-Migration
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Dependencies**: pandas, numpy, scikit-learn, xgboost, matplotlib, scipy

## Usage

CARS-M works with the following datasets:

* `predictive_maintenance.csv` – Real-world hybrid cloud-edge dataset (AI41, [Kaggle](https://www.kaggle.com/datasets))
* `FD001` & `FD003` – NASA Turbofan Engine Degradation Simulation datasets

Example usage:

```python
from carsm_final import run_carsm

# Load your dataset
X, y = load_dataset("predictive_maintenance.csv")

# Train/test split, scaling, and model fitting
results = run_carsm(X, y)

# Display metrics
print(results)
```

> For NASA datasets, similar preprocessing can be applied, with the same `run_carsm` function for evaluation.

## Repository Structure

```
Cluster-Aware-Risk-Scoring-and-Migration/
├─ carsm_final.py          # Final CARS-M implementation
├─ train_test_batch.py     # Original train/test split experiments
├─ basic_comps.py          # Early experiments with baseline vs CARS-M
├─ requirements.txt        # Python dependencies
├─ predictive_maintenance.csv  # Kaggle AI41 dataset 
├─ NASA_FD001/FD003        # NASA Turbofan datasets 
└─ README.md               # This file
```

## Results

CARS-M delivers statistically grounded, risk-informed predictive maintenance:

* **Hybrid cloud-edge dataset (AI41)**

  * Best Threshold: 0.85
  * Precision: 0.772
  * Recall: 0.647
  * Accuracy: 0.982
  * F1 Score: 0.704

* **NASA FD001/FD003 datasets**

  * CARS-M improves recall of vulnerable components, while maintaining high F1 scores and accuracy.
  * Empirical evaluation shows reliable early detection under imbalanced or uncertain stress conditions.

> Overall, CARS-M prioritizes high-risk components and enables preventive measures, reducing the risk of cascading failures in cloud-edge systems.

## Contact

Developed by **Sam4907**

* GitHub: [https://github.com/Sam4907](https://github.com/Sam4907)
* Email: [samyasmin49@gmail.com](mailto:samyasmin49@gmail.com)
