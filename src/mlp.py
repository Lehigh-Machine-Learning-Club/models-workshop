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

# Short descriptions for the activation function gallery
ACTIVATION_DESCRIPTIONS = {
    'None (Linear)': 'Output equals input. No non-linearity — the network collapses to a single linear transformation regardless of depth.',
    'Step Function': 'Binary output: 0 or 1. The classic perceptron approach. Sharp threshold means no useful gradient for learning.',
    'Sigmoid': 'Smooth S-curve squashing output to (0, 1). Great for probabilities, but gradients vanish for extreme inputs.',
    'ReLU': 'Rectified Linear Unit: max(0, x). The modern default — fast to compute, avoids vanishing gradients for positive values.',
    'Tanh': 'Hyperbolic tangent: outputs in (-1, 1). Zero-centered, which helps with optimization compared to Sigmoid.'
}


class MLP:
    def __init__(self, layer_sizes=None, seed=42):
        if layer_sizes is None:
            layer_sizes = [2, 3, 1]
        self.layer_sizes = layer_sizes
        self.rng = np.random.default_rng(seed)
        self.init_weights()
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
        # Gradient cache (populated after backward pass)
        self._dW1 = None
        self._db1 = None
        self._dW2 = None
        self._db2 = None
        
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
        
        self._dW2 = np.dot(self.A1.T, dZ2) / m
        self._db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * d_act_fn(self.Z1)
        
        self._dW1 = np.dot(self.X_cache.T, dZ1) / m
        self._db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= lr * self._dW1
        self.b1 -= lr * self._db1
        self.W2 -= lr * self._dW2
        self.b2 -= lr * self._db2
        
    def compute_loss_accuracy(self, Y):
        # Stable Binary Cross Entropy
        eps = 1e-15
        A2_clipped = np.clip(self.A2, eps, 1 - eps)
        loss = -np.mean(Y * np.log(A2_clipped) + (1 - Y) * np.log(1 - A2_clipped))
        
        predictions = (self.A2 > 0.5).astype(float)
        accuracy = np.mean(predictions == Y)
        return float(loss), float(accuracy)

    def train_step(self, X, Y, activation_name='Sigmoid', lr=0.1):
        """
        Combined forward + backward + history append.
        Returns (loss, accuracy) for this step.
        """
        self.forward(X, activation_name)
        loss, acc = self.compute_loss_accuracy(Y)
        self.backward(Y, activation_name, lr)
        self.loss_history.append(loss)
        self.accuracy_history.append(acc)
        return loss, acc

    def predict_single(self, x, activation_name='Sigmoid'):
        """
        Forward pass for a single sample. Returns all intermediate values.
        
        Args:
            x: shape (2,) or (1, 2) — single input sample
            activation_name: activation function name
            
        Returns:
            dict with keys: 'input', 'Z1', 'A1', 'Z2', 'A2'
        """
        act_fn, _ = ACTIVATIONS[activation_name]
        x = np.atleast_2d(x)
        
        Z1 = np.dot(x, self.W1) + self.b1      # (1, 3)
        A1 = act_fn(Z1)                          # (1, 3)
        Z2 = np.dot(A1, self.W2) + self.b2      # (1, 1)
        A2 = sigmoid(Z2)                          # (1, 1)
        
        return {
            'input': x.flatten(),
            'Z1': Z1.flatten(),
            'A1': A1.flatten(),
            'Z2': Z2.flatten(),
            'A2': A2.flatten()
        }

    def get_gradients(self):
        """
        Returns the most recently computed gradients.
        Call after backward().
        
        Returns:
            dict with keys: 'dW1', 'db1', 'dW2', 'db2' (or None if backward hasn't been called)
        """
        return {
            'dW1': self._dW1,
            'db1': self._db1,
            'dW2': self._dW2,
            'db2': self._db2
        }

    def get_neuron_boundaries(self, activation_name='Sigmoid'):
        """
        For each hidden neuron, compute the decision line: W1[0,j]*x + W1[1,j]*y + b1[0,j] = 0
        Returns a list of dicts with slope/intercept for each hidden neuron's boundary.
        
        This is the line where the pre-activation Z1[j] = 0, i.e., the threshold
        before the activation function fires.
        """
        boundaries = []
        for j in range(self.layer_sizes[1]):
            w0 = self.W1[0, j]  # weight for feature 0 (spikes)
            w1 = self.W1[1, j]  # weight for feature 1 (spots)
            b = self.b1[0, j]
            
            if abs(w1) > 1e-10:
                # y = -(w0/w1)*x - (b/w1)
                slope = -w0 / w1
                intercept = -b / w1
                boundaries.append({
                    'neuron_idx': j,
                    'slope': slope,
                    'intercept': intercept,
                    'w0': w0,
                    'w1': w1,
                    'bias': b,
                    'type': 'line'
                })
            else:
                # Vertical line: x = -b / w0
                if abs(w0) > 1e-10:
                    x_intercept = -b / w0
                else:
                    x_intercept = 0.0
                boundaries.append({
                    'neuron_idx': j,
                    'x_intercept': x_intercept,
                    'w0': w0,
                    'w1': w1,
                    'bias': b,
                    'type': 'vertical'
                })
        return boundaries

    def get_parameter_summary(self):
        """
        Returns a structured dictionary of all parameters for UI display.
        """
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy(),
            'total_params': (self.W1.size + self.b1.size + self.W2.size + self.b2.size),
            'layer_sizes': self.layer_sizes
        }

    def get_neuron_labels(self):
        """
        Generates playful labels for hidden neurons based on their weight patterns.
        A neuron with high W1[0, j] is a "Spike Detector," high W1[1, j] is a "Spot Detector," etc.
        """
        labels = []
        nicknames = []
        for j in range(self.layer_sizes[1]):
            w_spike = abs(self.W1[0, j])
            w_spot = abs(self.W1[1, j])
            sign_spike = "+" if self.W1[0, j] > 0 else "−"
            sign_spot = "+" if self.W1[1, j] > 0 else "−"
            
            if w_spike > w_spot * 1.5:
                nickname = "🔺 The Spike Spotter"
                detail = f"Primarily responds to Spikiness ({sign_spike}{w_spike:.2f})"
            elif w_spot > w_spike * 1.5:
                nickname = "🔵 The Spot Scanner"
                detail = f"Primarily responds to Spottiness ({sign_spot}{w_spot:.2f})"
            else:
                nickname = "⚖️ The Combo Detector"
                detail = f"Responds to both features ({sign_spike}{w_spike:.2f} spike, {sign_spot}{w_spot:.2f} spot)"
            
            labels.append(detail)
            nicknames.append(nickname)
        
        return nicknames, labels
