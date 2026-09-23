# 🚗 Auto Insurance Churn Analysis & Prediction

> End-to-end data engineering and machine learning project predicting customer churn for an auto insurance portfolio using 1.68M+ customer records.

---

## 📌 Problem Statement

Customer churn is one of the highest-cost problems in insurance. Acquiring a new customer costs 5–7x more than retaining an existing one. This project identifies the key drivers of churn and builds a predictive model that enables proactive retention outreach — targeting the right customers before they leave.

---

## 📂 Dataset

| File | Description | Rows |
|---|---|---|
| `autoinsurance_churn.csv` | Main table — customer features + churn label | ~1.68M |
| `customer.csv` | Tenure, annual premium, date of birth | ~1.68M |
| `demographic.csv` | Income, marital status, home value, credit | ~1.68M |
| `address.csv` | Geographic data — city, state, lat/lon | ~1.68M |
| `termination.csv` | Account suspension dates for churned customers | ~100K |

> **Note:** Raw data files are excluded from this repository per `.gitignore`. Download the dataset from [Kaggle](https://www.kaggle.com/) and place CSVs in `data/raw/`.

---

## 🏗️ Project Structure

```
auto-insurance-churn/
│
├── data/
│   ├── raw/              # Original CSVs (not committed)
│   └── processed/        # Cleaned, feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature transforms
│   └── 03_modeling.ipynb             # ML models + SHAP
│
├── src/
│   ├── data_loader.py    # Load & merge all 5 tables
│   ├── features.py       # Feature engineering functions
│   └── evaluate.py       # Model evaluation utilities
│
├── outputs/
│   └── figures/          # Charts and plots
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Churn rate distribution (~12% baseline)
- Churn by demographics: age, income, marital status, home ownership
- Churn by policy features: tenure, annual premium, credit score
- Geographic churn patterns by state and county

### 2. Feature Engineering
- Tenure bucketing (new / growing / loyal / long-term)
- Income brackets and home value encoding
- Age group segmentation
- Binary encoding for categorical flags

### 3. Modeling
- **Baseline:** Logistic Regression with class weighting
- **Primary Model:** XGBoost Classifier with hyperparameter tuning
- **Imbalanced data handling:** SMOTE oversampling (~12% churn rate)
- **Evaluation:** ROC-AUC, Precision-Recall, F1 (churn class)

### 4. Explainability
- SHAP values for global feature importance
- Individual prediction explanations for high-risk customers

---

## 📊 Key Findings

> *(Updated after full model run — see `notebooks/03_modeling.ipynb`)*

| Driver | Impact |
|---|---|
| Days Tenure | Shorter tenure → higher churn risk |
| Annual Premium | Higher premiums correlate with increased churn |
| Good Credit | Good credit customers churn less |
| Age | Younger customers churn at higher rates |
| Home Ownership | Renters churn more than homeowners |

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.2-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.45-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-yellow)

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Narasimha3008/auto-insurance-churn.git
cd auto-insurance-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw CSVs in data/raw/

# 4. Run notebooks in order
jupyter lab
```

---

## 👤 Author

**Narasimha Naidu Kilari**  
Data Engineer | Data Analyst  
[LinkedIn](https://linkedin.com/in/narasimhanaidu-kilari) · [GitHub](https://github.com/Narasimha3008)

---

## 📄 License

MIT License — free to use and adapt with attribution.
