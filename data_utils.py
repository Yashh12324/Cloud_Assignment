import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def prepare_xy(df):
    # drop customer id
    df = df.copy()
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    # target
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
