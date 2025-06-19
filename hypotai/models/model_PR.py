# hypotai/model_PR.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def train_polynomial_model(data: pd.DataFrame, degree:int = 2, include_angles: bool = False):
    """
    Trains a linear regression model on triangle data.
    Adds polynomial features if specified.

    Parameters:
    - data: DataFrame with columns ['a', 'b'] and target 'c'
    - degree: degree of polynomial features to include
    - include_angles: whether to include angle features in the model

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
    if include_angles:
        X_train, angle_train = X_train[["a","b"]], X_train["angle_deg"].values.reshape(-1, 1)
        X_test, angle_test = X_test[["a","b"]], X_test["angle_deg"].values.reshape(-1, 1)
        # If angles are included, we need to transform the features
        X_train = PolynomialFeatures(degree=degree).fit_transform(X_train)
        X_test = PolynomialFeatures(degree=degree).fit_transform(X_test)
        # Re-adding angles to the transformed features
        X_train_poly = np.concatenate([X_train, angle_train], axis=1)
        X_test_poly = np.concatenate([X_test, angle_test], axis=1)
    else:
        # If angles are not included, we can directly use the features
        X_train_poly = PolynomialFeatures(degree=degree).fit_transform(X_train)
        X_test_poly = PolynomialFeatures(degree=degree).fit_transform(X_test)
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }

    return model, metrics


