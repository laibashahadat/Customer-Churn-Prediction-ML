from src.data_loader import build_synthetic_dataset
from src.data_preprocessing import clean_dataframe, prepare_training_data


def test_preprocessing_and_split():
    df = build_synthetic_dataset()
    cleaned = clean_dataframe(df)
    assert "churn" in cleaned.columns
    assert "customerid" not in cleaned.columns
    assert cleaned.shape[0] == df.shape[0]

    X_train, X_test, y_train, y_test = prepare_training_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert y_train.nunique() >= 2
    assert y_test.nunique() >= 1
