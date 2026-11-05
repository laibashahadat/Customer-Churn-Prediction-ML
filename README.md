# Customer Churn Prediction & Analytics System

A production-style churn prediction project built with Python, scikit-learn, Flask, and a responsive analytics dashboard. This project is designed as a portfolio-ready machine learning application that can ingest the IBM Telco Customer Churn dataset, compare multiple classification models, and present results through a web dashboard and REST API.

## Problem Statement

Customer churn can significantly reduce recurring revenue and increase customer acquisition costs. Organizations need a way to identify likely churners early, prioritize retention outreach, and monitor churn patterns by customer segment.

## Objectives

- Predict churn using historical customer data.
- Compare multiple machine learning models.
- Evaluate performance with standard classification metrics.
- Deliver individual and batch predictions.
- Store prediction history locally with SQLite.
- Present the system in a professional dashboard.

## Features

- Data ingestion and validation for the IBM Telco dataset.
- End-to-end preprocessing with missing value handling and classification-safe encoding.
- Train/test split with stratification and balanced modeling.
- Comparison of Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost (when available).
- Model evaluation with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Individual prediction form and CSV batch prediction flow.
- Risk classification for low/medium/high churn risk.
- SQLite history tracking for predictions.
- Flask REST API and SaaS-style dashboard UI.

## Technology Stack

- Python 3.11
- pandas, NumPy, scikit-learn
- seaborn, matplotlib
- xgboost
- Flask + Flask-CORS
- SQLite
- Joblib
- Pytest

## Dataset Source

The project is designed for the IBM Telco Customer Churn dataset from Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Important: the dataset is not bundled in this repository. Download it and place the CSV file at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

If the file is absent, the app falls back to a small synthetic demo dataset for smoke testing, but the real IBM dataset should be used for serious analysis.

## Architecture Overview

```text
customer-churn-prediction/
├── app/
│   ├── app.py
│   ├── database.py
│   ├── static/
│   └── templates/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── predict.py
│   ├── train.py
│   └── __init__.py
├── tests/
├── config.py
├── requirements.txt
├── run.py
├── README.md
└── .gitignore
```

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Data Preparation

Download the IBM Telco dataset and store it in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

To train the model:

```bash
python -c "from src.data_loader import load_churn_data; from src.train import train_and_compare_models; df = load_churn_data(); train_and_compare_models(df)"
```

## Run the App

```bash
python run.py
```

Then open:

http://127.0.0.1:5000

## API Overview

### Health check
```bash
GET /api/health
```

### Dashboard summary
```bash
GET /api/dashboard
```

### Single prediction
```bash
POST /api/predict
```

Request JSON example:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 80.5,
  "TotalCharges": 950.0
}
```

### Batch prediction
```bash
POST /api/batch-predict
```
Upload a CSV file with the same customer feature columns used by the Telco dataset.

### Prediction history
```bash
GET /api/history
```

### Model performance
```bash
GET /api/model-performance
```

## Model Evaluation Notes

This project evaluates models using relevant churn metrics rather than relying on accuracy alone. In churn problems, recall is often critical because missing a likely churner can have a larger business cost than a false positive. Precision remains important when retention teams have limited capacity and cannot contact every customer flagged as at risk.

## Testing

```bash
pytest -q
```

## Limitations

- The project relies on a real dataset for robust business evaluation; the synthetic fallback is only for local smoke testing.
- The dashboard is a clean implementation for portfolio use but is not a replacement for a production-grade analytics platform.
- Model interpretability is provided through model-supported feature importance and permutation importance, which are useful but not a substitute for domain analysis.

## Future Improvements

- Add cross-validation tuning for model selection.
- Introduce SHAP-based explainability for deeper insight.
- Improve dashboard charts with richer segmentation views.
- Add authentication and user management for multi-user deployments.
- Package the app for deployment with Docker or a cloud platform.

## License

This project is intended for educational and portfolio use. Use the IBM Telco dataset according to its licensing and Kaggle usage terms.
