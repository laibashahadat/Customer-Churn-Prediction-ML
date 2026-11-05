from pathlib import Path

from config import MODEL_PATH, MODEL_SUMMARY_PATH
from src.data_loader import build_synthetic_dataset
from src.train import train_and_compare_models


def test_model_training_creates_artifacts():
    summary, results = train_and_compare_models(build_synthetic_dataset())
    assert summary["best_model"]
    assert isinstance(results, dict)
    assert MODEL_PATH.exists()
    assert MODEL_SUMMARY_PATH.exists()
