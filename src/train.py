import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_DIR, MODEL_PATH, MODEL_SUMMARY_PATH, TARGET_COLUMN
from src.data_preprocessing import build_preprocessor, clean_dataframe, prepare_training_data

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


def build_model_suite():
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    return models


def fit_model(model_name, model, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def train_and_compare_models(df):
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    model_suite = build_model_suite()
    results = {}
    for name, model in model_suite.items():
        pipeline = fit_model(name, model, X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = {
            "metrics": metrics,
            "pipeline": pipeline,
        }
    best_name, best_entry = max(results.items(), key=lambda item: (item[1]["metrics"]["roc_auc"], item[1]["metrics"]["recall"], item[1]["metrics"]["f1"]))
    summary = {
        "best_model": best_name,
        "results": {key: value["metrics"] for key, value in results.items()},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_entry["pipeline"], MODEL_PATH)
    with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary, results


def load_saved_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model has been saved yet. Train the model first.")
    return joblib.load(MODEL_PATH)


def get_best_model_name():
    if not MODEL_SUMMARY_PATH.exists():
        return None
    with open(MODEL_SUMMARY_PATH, "r", encoding="utf-8") as file:
        summary = json.load(file)
    return summary.get("best_model")
