import pandas as pd

from config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH


REQUIRED_COLUMNS = {
    "customerID",
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
    "Churn",
}


def load_churn_data(dataset_path=None):
    """Load the Telco customer churn dataset from a CSV path."""
    path = dataset_path or RAW_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco dataset and place it in data/raw/."
        )
    df = pd.read_csv(path)
    missing_required = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")
    return df


def load_or_create_processed_data():
    """Return a processed dataset if it exists, otherwise load raw data."""
    if PROCESSED_DATASET_PATH.exists():
        return pd.read_csv(PROCESSED_DATASET_PATH)
    return load_churn_data()


def build_synthetic_dataset():
    """Create a compact, realistic synthetic dataset for tests and smoke checks."""
    records = []
    for idx in range(1, 201):
        tenure = (idx % 72) + 1
        monthly = 35 + (idx % 60)
        total = monthly * tenure
        churn = 1 if ((idx % 5) == 0 or tenure < 12 and idx % 3 == 0) else 0
        record = {
            "customerID": f"CUST_{idx:04d}",
            "gender": "Female" if idx % 2 == 0 else "Male",
            "SeniorCitizen": 1 if idx % 10 == 0 else 0,
            "Partner": "Yes" if idx % 2 == 0 else "No",
            "Dependents": "Yes" if idx % 4 == 0 else "No",
            "tenure": tenure,
            "PhoneService": "Yes",
            "InternetService": ["DSL", "Fiber optic", "No"][idx % 3],
            "OnlineSecurity": ["No", "Yes", "No"][idx % 3],
            "OnlineBackup": ["No", "Yes", "No"][idx % 3],
            "DeviceProtection": ["No", "Yes", "No"][idx % 3],
            "TechSupport": ["No", "Yes", "No"][idx % 3],
            "Contract": ["Month-to-month", "One year", "Two year"][idx % 3],
            "PaperlessBilling": "Yes" if idx % 2 == 0 else "No",
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"][idx % 4],
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": "Yes" if churn else "No",
        }
        records.append(record)
    return pd.DataFrame(records)
