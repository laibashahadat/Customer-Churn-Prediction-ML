import json
from pathlib import Path

import pandas as pd

from config import MODEL_PATH, RISK_THRESHOLDS, TARGET_COLUMN
from src.data_preprocessing import clean_dataframe
from src.train import load_saved_model


REQUIRED_RECORD_FIELDS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]


def _normalize_record(record):
    cleaned = {}
    for key, value in record.items():
        cleaned[str(key).strip()] = value
    normalized = {str(key).strip().lower(): value for key, value in cleaned.items()}
    for field in REQUIRED_RECORD_FIELDS:
        if field not in normalized:
            raise ValueError(f"Missing required field: {field}")
    frame = pd.DataFrame([normalized])
    frame = clean_dataframe(frame)
    return frame


def classify_risk(probability):
    if probability >= RISK_THRESHOLDS["medium"]:
        return "High risk"
    if probability >= RISK_THRESHOLDS["low"]:
        return "Medium risk"
    return "Low risk"


def predict_single_customer(record):
    model = load_saved_model()
    frame = _normalize_record(record)
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = "Yes" if probability >= 0.5 else "No"
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": classify_risk(probability),
        "threshold": 0.5,
    }


def predict_batch_dataframe(df):
    model = load_saved_model()
    cleaned = clean_dataframe(df)
    missing = [column for column in REQUIRED_RECORD_FIELDS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Batch file is missing required fields: {missing}")
    predictions = model.predict_proba(cleaned[REQUIRED_RECORD_FIELDS])
    result = cleaned.copy()
    result["predicted_churn"] = ["Yes" if value >= 0.5 else "No" for value in predictions[:, 1]]
    result["churn_probability"] = predictions[:, 1]
    result["risk_category"] = [classify_risk(value) for value in predictions[:, 1]]
    return result
