<div align="center">

# 📊 Customer Churn Prediction Framework

**A Cost-Sensitive and Explainable ML System for Telecom Customer Retention**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006E96?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=flat-square)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

</div>

---

## What This Project Does

Predicts which telecom customers are about to cancel their subscription — **before they leave** — using classical machine learning. Goes beyond just prediction by answering two questions most ML projects ignore:

- **Which threshold actually saves the most money?** (not just the default 0.50)
- **Why is the model predicting churn, and what should the business do about it?**

---

## Three Contributions

| # | What | Result |
|---|------|--------|
| 1 | Benchmark 5 models × 5 imbalance strategies = 25 experiments, statistically validated | SMOTEENN + XGBoost wins — CV F1 = **0.621 ± 0.012** (p = 0.031) |
| 2 | Cost-sensitive threshold sweep (10:1 asymmetric cost matrix) | Optimal t* = **0.06** → **26.4% cost reduction**, missed churners down **57.7%** |
| 3 | SHAP explainability mapped to 8 specific retention actions | Top driver: Contract type (SHAP = **1.276**) |

---

## Key Results

```
Dataset  : IBM Telco Customer Churn (7,043 customers · 26.5% churn rate)
Best model: SMOTEENN + XGBoost (tuned)

At default threshold (0.50) → Recall: 0.810 | F1: 0.609 | Cost: $1,028
At optimal threshold (0.06) → Recall: 0.920 | F1: 0.586 | Cost:   $757  ✅ saves 26.4%
```

---

## Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `imbalanced-learn` `SHAP` `pandas` `matplotlib` `scipy` `Google Colab`

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AdiRatnam/customer-churn-prediction.git

# 2. Install dependencies
pip install scikit-learn xgboost lightgbm imbalanced-learn shap pandas numpy matplotlib seaborn scipy

# 3. Download dataset from Kaggle
#    kaggle.com/datasets/blastchar/telco-customer-churn
#    Place CSV in project root

# 4. Open notebook
jupyter notebook Churn_Prediction_Notebook.ipynb
```

> All experiments use `random_state=42` — fully reproducible.

---

## Dataset

IBM Telco Customer Churn — [Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> Dataset not included in this repo. Download from Kaggle and place in root directory.

---


## License

[MIT](LICENSE) © 2026 Adi Ratnam
