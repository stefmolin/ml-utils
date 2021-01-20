"""Utilities for evaluating regression models."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def adjusted_r2(model, X, y):
    """
    Calculate the adjusted R^2.

    Parameters:
        - model: Estimator object with a `predict()` method
        - X: The values to use for prediction.
        - y: The true values for scoring.

    Returns:
        The adjusted R^2 score.
    """
    r2 = r2_score(y, model.predict(X))
    n_obs, n_regressors = X.shape
    adj_r2 = 1 - (1 - r2) * (n_obs - 1)/(n_obs - n_regressors - 1)
    return adj_r2

def plot_residuals(y_test, preds):
    """
    Plot residuals to evaluate regression.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y

    Returns:
        Subplots of residual scatter plot and
        residual KDE plot.
    """
    residuals = y_test - preds

    fig, axes = plt.subplots(1, 2, figsize=(15, 3))

    axes[0].scatter(np.arange(residuals.shape[0]), residuals)
    axes[0].set(xlabel='Observation', ylabel='Residual')

    residuals.plot(kind='kde', ax=axes[1])
    axes[1].set_xlabel('Residual')

    plt.suptitle('Residuals')
    return axes
