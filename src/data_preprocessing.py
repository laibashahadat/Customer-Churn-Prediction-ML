import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PROCESSED_DATASET_PATH, TARGET_COLUMN


CATEGORICAL_COLUMNS = [
    "gender",
    "partner",
    "dependents",
    "phoneservice",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]
NUMERIC_COLUMNS = [
    "seniorcitizen",
    "tenure",
    "monthlycharges",
    "totalcharges",
]


def clean_dataframe(df):
    """Normalize the input data to prepare it for training."""
    working = df.copy()
    if "customerID" in working.columns:
        working = working.drop(columns=["customerID"])
    working.columns = [str(column).strip().lower() for column in working.columns]
    if "totalcharges" in working.columns:
        working["totalcharges"] = pd.to_numeric(working["totalcharges"], errors="coerce")
    if "monthlycharges" in working.columns:
        working["monthlycharges"] = pd.to_numeric(working["monthlycharges"], errors="coerce")
    if "seniorcitizen" in working.columns:
        working["seniorcitizen"] = working["seniorcitizen"].astype(int)
    for column in working.columns:
        if working[column].dtype == "object":
            working[column] = working[column].astype(str).str.strip()
            working[column] = working[column].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    if "churn" in working.columns:
        working["churn"] = working["churn"].astype(str).str.strip().str.lower().replace({"yes": "Yes", "no": "No"})
    working = working.drop_duplicates()
    return working


def get_feature_columns(df):
    features = [column for column in df.columns if column != TARGET_COLUMN.lower()]
    return features


def build_preprocessor(X):
    numeric_features = [column for column in NUMERIC_COLUMNS if column in X.columns]
    categorical_features = [column for column in CATEGORICAL_COLUMNS if column in X.columns]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_training_data(df):
    """Return cleaned features and target with train/test split."""
    working = clean_dataframe(df)
    if "churn" not in working.columns:
        raise ValueError("The supplied dataframe must include a Churn column.")
    target = working["churn"].copy().map({"Yes": 1, "No": 0})
    features = working.drop(columns=["churn"])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def get_training_schema():
    schema = {
        "required": [
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
        ],
        "optional": ["customerID"],
    }
    return schema


def save_processed_dataset(df):
    output_path = Path(PROCESSED_DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_processed_dataset():
    if Path(PROCESSED_DATASET_PATH).exists():
        return pd.read_csv(PROCESSED_DATASET_PATH)
    return None


def get_preprocessing_summary(df):
    working = clean_dataframe(df)
    summary = {
        "rows": int(len(working)),
        "columns": list(working.columns),
        "missing_values": {column: int(working[column].isna().sum()) for column in working.columns},
        "duplicated_rows": int(working.duplicated().sum()),
    }
    return summary
