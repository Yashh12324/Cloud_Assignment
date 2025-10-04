# In train.py
import argparse
import mlflow
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Helper Functions ---
def load_data(path: str):
    df = pd.read_csv(path)
    return df

def prepare_xy(df):
    df = df.copy()
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def build_pipeline():
    numeric_features = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
    categorical_features = ['Gender','Subscription Type','Contract Length']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])
    return clf

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1]
    metrics = {
        'accuracy': float(accuracy_score(y_val, preds)),
        'precision': float(precision_score(y_val, preds, zero_division=0)),
        'recall': float(recall_score(y_val, preds, zero_division=0)),
        'f1': float(f1_score(y_val, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_val, probs))
    }
    return metrics

# --- Main Execution ---
def main(args):
    mlflow.autolog(log_models=True, exclusive=True)
    df = load_data(args.data_path)
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    pipeline = build_pipeline()
    param_grid = { 'classifier_n_estimators': [50, 100], 'classifier_max_depth': [None, 10] }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    metrics = evaluate_model(best, X_val, y_val)
    print("Best params:", grid.best_params_)
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data CSV")
    args = parser.parse_args()
    main(args)
