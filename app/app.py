import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from config import BASE_DIR, MODEL_PATH, RAW_DATASET_PATH
from src.data_loader import build_synthetic_dataset
from src.data_preprocessing import clean_dataframe
from src.explainability import summarize_explanations
from src.predict import classify_risk, predict_batch_dataframe, predict_single_customer
from src.train import get_best_model_name, load_saved_model, train_and_compare_models
from app.database import get_prediction_history, init_db, save_prediction_record


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


def _load_dashboard_data():
    if RAW_DATASET_PATH.exists():
        df = pd.read_csv(RAW_DATASET_PATH)
    else:
        df = build_synthetic_dataset()
    cleaned = clean_dataframe(df)
    if "churn" in cleaned.columns:
        cleaned["churn"] = cleaned["churn"].str.lower().replace({"yes": "Yes", "no": "No"})
    return cleaned


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


@app.route("/model-performance")
def model_performance_page():
    return render_template("model_performance.html")


@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.route("/batch-prediction")
def batch_prediction_page():
    return render_template("batch_prediction.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_exists": MODEL_PATH.exists(), "dataset_ready": RAW_DATASET_PATH.exists()})


@app.route("/api/dashboard")
def dashboard_api():
    df = _load_dashboard_data()
    data = {"total_customers": int(len(df)), "churned_customers": int((df["churn"] == "Yes").sum()), "active_customers": int((df["churn"] == "No").sum())}
    data["churn_rate"] = round(data["churned_customers"] / max(1, data["total_customers"]), 4)
    high_risk = 0
    if MODEL_PATH.exists():
        try:
            model = load_saved_model()
            proba = model.predict_proba(df.drop(columns=["churn"]))[:, 1]
            high_risk = int((proba >= 0.6).sum())
        except Exception:
            high_risk = 0
    data["high_risk_customers"] = high_risk
    data["distribution"] = {
        "Yes": int((df["churn"] == "Yes").sum()),
        "No": int((df["churn"] == "No").sum()),
    }
    return jsonify(data)


@app.route("/api/model-performance")
def model_performance_api():
    if not MODEL_PATH.exists():
        return jsonify({"message": "No trained model available yet. Run the training pipeline first."}), 404
    try:
        summary = {}
        with open(BASE_DIR / "models" / "model_summary.json", "r", encoding="utf-8") as file:
            summary = json.load(file)
        return jsonify(summary)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"]) 
def predict_api():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400
    try:
        result = predict_single_customer(payload)
        save_prediction_record(payload, result["prediction"], result["probability"], result["risk_category"])
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "The model has not been trained yet. Please train the model before making predictions."}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/batch-predict", methods=["POST"]) 
def batch_predict_api():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No CSV file was uploaded."}), 400
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400
    try:
        df = pd.read_csv(file)
        result = predict_batch_dataframe(df)
        return jsonify({
            "rows_processed": int(len(result)),
            "summary": {
                "risk_breakdown": result["risk_category"].value_counts().to_dict(),
                "predicted_yes": int((result["predicted_churn"] == "Yes").sum()),
                "predicted_no": int((result["predicted_churn"] == "No").sum()),
            },
            "records": result[["predicted_churn", "churn_probability", "risk_category"]].to_dict(orient="records"),
        })
    except Exception as exc:
        return jsonify({"error": f"Unable to process file: {exc}"}), 400


@app.route("/api/history")
def history_api():
    records = get_prediction_history(limit=30)
    return jsonify({"history": records})


@app.route("/api/train")
def train_api():
    if not RAW_DATASET_PATH.exists():
        df = build_synthetic_dataset()
    else:
        df = pd.read_csv(RAW_DATASET_PATH)
    summary, _ = train_and_compare_models(df)
    return jsonify(summary)


init_db()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
