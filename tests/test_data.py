import pandas as pd
from src.data import preprocess_data, split_data

def test_no_nan():
    df = pd.read_csv("data/creditcard.csv")
    df = preprocess_data(df)
    assert df.isnull().sum().sum() == 0

def test_split():
    df = pd.read_csv("data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > len(X_test)

def test_dimensions():
    df = pd.read_csv("data/creditcard.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    assert X_train.shape[0] == y_train.shape[0]