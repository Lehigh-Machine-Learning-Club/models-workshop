import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression


def get_poisonous_fruit_data(n_samples=200, noise=0.15, random_state=42):
    """
    Generates a synthetic binary classification dataset for "Poisonous Fruits".
    We use the 'moons' dataset from scikit-learn as proxy for a non-linear problem.
    The features are normalized to roughly [0, 1] range to represent 'Spikes' and 'Spots'.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Normalize features to roughly [0, 1] for interpreting as Fruit traits
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = y.reshape(-1, 1)  # Ensure Y is shape (N, 1)
    return X, y


def get_linear_baseline_accuracy(X, y):
    """
    Fits a Logistic Regression (linear boundary) on the data and returns
    the training accuracy. This demonstrates the ceiling of linear models
    on non-linearly separable data.
    
    Returns:
        accuracy (float): Training accuracy of the linear model (0 to 1)
        model: The fitted LogisticRegression model for optional visualization
    """
    y_flat = y.flatten()
    model = LogisticRegression(max_iter=200)
    model.fit(X, y_flat)
    accuracy = model.score(X, y_flat)
    return accuracy, model


def get_sample_by_index(X, y, idx):
    """
    Returns a single sample and its label by index.
    
    Args:
        X: feature array (N, 2)
        y: label array (N, 1)
        idx: integer index
        
    Returns:
        x_sample (1, 2), y_label (scalar)
    """
    return X[idx:idx+1], y[idx, 0]
