from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_importance(model_pipeline, feature_names):
    """Extract feature importance from the trained model when supported."""
    model = model_pipeline.named_steps["model"]
    importance_data = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_data = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
        importance_data = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(importances),
        }).sort_values("importance", ascending=False)
    return importance_data


def permutation_feature_importance(model_pipeline, X_test, y_test, n_repeats=10):
    """Estimate permutation importance for model explanation."""
    result = permutation_importance(model_pipeline, X_test, y_test, n_repeats=n_repeats, random_state=42)
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


def summarize_explanations(model_pipeline, X_test, y_test):
    features = X_test.columns.tolist()
    importance = get_feature_importance(model_pipeline, features)
    permutation_scores = permutation_feature_importance(model_pipeline, X_test, y_test)
    return {
        "feature_importance": importance.to_dict(orient="records") if importance is not None else [],
        "permutation_importance": permutation_scores.to_dict(orient="records"),
    }
