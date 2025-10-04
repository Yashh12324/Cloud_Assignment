# in tests/test_logic.py
import pandas as pd
import pytest
from train import prepare_xy

def test_prepare_xy():
    data = {'CustomerID': [1], 'Churn': [0], 'FeatureA': [100]}
    df = pd.DataFrame(data)
    X, y = prepare_xy(df)
    assert 'CustomerID' not in X.columns