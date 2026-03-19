import numpy as np

# --- Modular Activation Functions ---
def linear(x): return x
def d_linear(x): return np.ones_like(x)

def step(x): return np.where(x > 0, 1.0, 0.0)
def d_step(x): return np.zeros_like(x)  # Derivative is technically undefined at 0, 0 elsewhere

def sigmoid(x): 
    # Clip for stability
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))
def d_sigmoid(x): 
    s = sigmoid(x)
    return s * (1 - s)

def relu(x): return np.maximum(0, x)
def d_relu(x): return np.where(x > 0, 1.0, 0.0)

def tanh_act(x): return np.tanh(x)
def d_tanh(x): return 1.0 - np.tanh(x)**2

ACTIVATIONS = {
    'None (Linear)': (linear, d_linear),
    'Step Function': (step, d_step),
    'Sigmoid': (sigmoid, d_sigmoid),
    'ReLU': (relu, d_relu),
    'Tanh': (tanh_act, d_tanh)
}

class MLP:
    def __init__(self, layer_sizes=[2, 3, 1], seed=42):
        self.layer_sizes = layer_sizes
        self.rng = np.random.default_rng(seed)
        self.init_weights()
        
    def init_weights(self):
        # Initialize randomly with slightly scaled values
        # W1: 2 x 3, b1: 1 x 3
        self.W1 = self.rng.standard_normal((self.layer_sizes[0], self.layer_sizes[1])) * 0.5
        self.b1 = np.zeros((1, self.layer_sizes[1]))
        
        # W2: 3 x 1, b2: 1 x 1
        self.W2 = self.rng.standard_normal((self.layer_sizes[1], self.layer_sizes[2])) * 0.5
        self.b2 = np.zeros((1, self.layer_sizes[2]))
        
    def forward(self, X, activation_name='Sigmoid'):
        act_fn, _ = ACTIVATIONS[activation_name]
        
        self.X_cache = np.array(X)
        
        # Layer 1
        self.Z1 = np.dot(self.X_cache, self.W1) + self.b1
        self.A1 = act_fn(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)  # Output layer always uses Sigmoid for classification probabilities
        
        return self.A2

    def backward(self, Y, activation_name='Sigmoid', lr=0.1):
        _, d_act_fn = ACTIVATIONS[activation_name]
        m = Y.shape[0]
        
        # Binary Cross Entropy Derivative for Sigmoid Output Layer
        # dZ2 = A2 - Y
        dZ2 = self.A2 - Y 
        
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * d_act_fn(self.Z1)
        
        dW1 = np.dot(self.X_cache.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        
    def compute_loss_accuracy(self, Y):
        # Stable Binary Cross Entropy
        eps = 1e-15
        A2_clipped = np.clip(self.A2, eps, 1 - eps)
        loss = -np.mean(Y * np.log(A2_clipped) + (1 - Y) * np.log(1 - A2_clipped))
        
        predictions = (self.A2 > 0.5).astype(float)
        accuracy = np.mean(predictions == Y)
        return float(loss), float(accuracy)
