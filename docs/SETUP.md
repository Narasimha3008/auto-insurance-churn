# Setup & GitHub Push Guide

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Place your data

Download the dataset and place all 5 CSV files in `data/raw/`:
- autoinsurance_churn.csv
- customer.csv
- demographic.csv
- address.csv
- termination.csv

## 3. Run notebooks in order

```bash
jupyter lab
```

Open and run:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_modeling.ipynb`

## 4. Push to GitHub

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit: auto insurance churn analysis project"

# Create repo on github.com, then:
git remote add origin https://github.com/Narasimha3008/auto-insurance-churn.git
git branch -M main
git push -u origin main
```

## 5. Recommended GitHub repo settings

- Add description: "End-to-end churn analysis and ML prediction on 1.68M auto insurance records"
- Add topics: `python`, `machine-learning`, `churn-prediction`, `xgboost`, `shap`, `data-science`
- Pin it to your profile
