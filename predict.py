import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ""
MODEL_PATH = ""

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_single(model, input_dict):
    # expects dict mapping of features (same names as training X columns)
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0,1] if hasattr(model, "predict_proba") else None
    return {'prediction': int(pred), 'probability': float(proba) if proba is not None else None}

if __name__ == "__main__":
    model = load_model()
    sample = {
        "Age": 40,
        "Gender": "Female",
        "Tenure": 20,
        "Usage Frequency": 10,
        "Support Calls": 3,
        "Payment Delay": 5,
        "Subscription Type": "Standard",
        "Contract Length": "Monthly",
        "Total Spend": 400,
        "Last Interaction": 5
    }
    print(predict_single(model, sample))
