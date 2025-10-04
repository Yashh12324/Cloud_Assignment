import joblib
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_utils import load_data, prepare_xy, train_val_split

#Download the data from this url - https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download&select=customer_churn_dataset-training-master.csv

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "" #set the datapath
MODEL_DIR = "model" #set the model path
MODEL_DIR.mkdir(exist_ok=True)

def build_pipeline():
    # feature groups (automatically map names present)
    numeric_features = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
    categorical_features = ['Gender','Subscription Type','Contract Length']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [c for c in numeric_features if c in numeric_features]),
            ('cat', categorical_transformer, [c for c in categorical_features if c in categorical_features])
        ], remainder='drop'
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return clf

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': float(accuracy_score(y_val, preds)),
        'precision': float(precision_score(y_val, preds, zero_division=0)),
        'recall': float(recall_score(y_val, preds, zero_division=0)),
        'f1': float(f1_score(y_val, preds, zero_division=0)),
    }
    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_val, probs))
        except Exception:
            metrics['roc_auc'] = None
    return metrics

def main():
    df = load_data(DATA_PATH)
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)

    pipeline = build_pipeline()

    # quick hyperparameter sweep (small grid for demo)
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    metrics = evaluate_model(best, X_val, y_val)
    print("Best params:", grid.best_params_)
    print("Validation metrics:", metrics)

    # save model + metadata
    model_path = MODEL_DIR / "churn_rf.joblib"
    joblib.dump(best, model_path)

    meta = {
        'model_path': str(model_path),
        'best_params': grid.best_params_,
        'metrics': metrics
    }
    with open(MODEL_DIR / "model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
