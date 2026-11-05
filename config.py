from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "app" / "prediction_history.db"

RAW_DATASET_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "processed_churn_data.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.json"

RISK_THRESHOLDS = {
    "low": 0.35,
    "medium": 0.6,
}

TARGET_COLUMN = "Churn"
PREDICTOR_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
