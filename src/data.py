import numpy as np
from sklearn.datasets import make_moons

def get_poisonous_fruit_data(n_samples=200, noise=0.15, random_state=42):
    """
    Generates a synthetic binary classification dataset for "Poisonous Fruits".
    We use the 'moons' dataset from scikit-learn as proxy for a non-linear problem.
    The features are normalized to roughly [0, 1] range to represent 'Spikes' and 'Spots'.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Normalize features to roughly [0, 1] for interpreting as Fruit traits
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = y.reshape(-1, 1) # Ensure Y is shape (N, 1)
    return X, y
