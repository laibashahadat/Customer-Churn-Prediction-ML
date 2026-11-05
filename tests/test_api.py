import io

from app.app import app
from src.data_loader import build_synthetic_dataset
from src.train import train_and_compare_models


def test_api_health():
    with app.test_client() as client:
        response = client.get('/api/health')
        assert response.status_code == 200


def test_single_prediction_endpoint():
    train_and_compare_models(build_synthetic_dataset())
    payload = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.5,
        'TotalCharges': 780.0,
    }
    with app.test_client() as client:
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'probability' in data


def test_batch_prediction_endpoint():
    train_and_compare_models(build_synthetic_dataset())
    csv = b'gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\nMale,0,Yes,No,10,Yes,Fiber optic,No,Yes,No,No,Month-to-month,Yes,Electronic check,70.5,705.0,No\nFemale,1,No,No,3,Yes,DSL,No,No,No,No,Month-to-month,No,Mailed check,55.0,165.0,Yes\n'
    with app.test_client() as client:
        response = client.post('/api/batch-predict', data={'file': (io.BytesIO(csv), 'customers.csv')})
        assert response.status_code == 200
        data = response.get_json()
        assert data['rows_processed'] >= 1
