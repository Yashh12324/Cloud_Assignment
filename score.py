import os
import joblib
import json
import pandas as pd

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl") # MLflow saves as model.pkl
    model = joblib.load(model_path)

def run(raw_data):
    try:
        input_dict = json.loads(raw_data)
        df = pd.DataFrame([input_dict])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0,1]
        result = {'prediction': int(pred), 'probability': float(proba)}
        return result
    except Exception as e:
        return str(e)
