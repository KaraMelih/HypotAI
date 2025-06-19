# hypotai/model_LR.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def train_linear_model(data: pd.DataFrame, include_angles: bool = False):
    """
    Trains a linear regression model on triangle data.

    Parameters:
    - data: DataFrame with columns ['a', 'b'] and target 'c'

    Returns:
    - model: trained LinearRegression model
    - metrics: dict of training performance
    """
    if include_angles:
        X = data[["a", "b", "angle_deg"]]
    else:
        X = data[["a", "b"]]
    y = data["c"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    return model, metrics

def cross_validate_model(data: pd.DataFrame, cv: int = 5):
    X = data[["a", "b"]]
    y = data["c"]

    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    return scores
