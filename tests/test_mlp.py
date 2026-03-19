import numpy as np
from src.mlp import MLP

def test_mlp_forward_backward():
    mlp = MLP(layer_sizes=[2, 3, 1], seed=42)
    X = np.array([[0.1, 0.2], [0.8, 0.9]])
    Y = np.array([[0.0], [1.0]])
    
    # Forward pass
    output = mlp.forward(X, activation_name='Sigmoid')
    assert output.shape == (2, 1)
    assert np.all(output >= 0) and np.all(output <= 1)
    
    # Capture old weights
    old_W1 = mlp.W1.copy()
    old_W2 = mlp.W2.copy()
    
    # Backward pass
    mlp.backward(Y, activation_name='Sigmoid', lr=0.1)
    
    # Assert weights updated (and gradients flowed properly)
    assert not np.allclose(old_W1, mlp.W1)
    assert not np.allclose(old_W2, mlp.W2)

def test_activations_dynamic():
    mlp = MLP()
    X = np.array([[0.5, -0.5]])
    
    # Ensure hot-swapping activations yields different intermediates
    mlp.forward(X, activation_name='ReLU')
    a1_relu = mlp.A1.copy()
    
    mlp.forward(X, activation_name='Sigmoid')
    a1_sigmoid = mlp.A1.copy()
    
    assert not np.allclose(a1_relu, a1_sigmoid)

