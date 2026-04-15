"""
MNIST MLP Models for Phase 2.
Provides both single-hidden-layer and dual-hidden-layer architectures,
with introspection methods for feature visualization and activation analysis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MNIST_MLP_Small(nn.Module):
    """Single hidden layer: 784 → 64 → 10"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.hidden_size = hidden_size
        self.architecture = f"784 → {hidden_size} → 10"
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_layer_activations(self, x):
        """Returns activations at every layer for a single input."""
        x = x.view(-1, 784)
        a0 = x.clone()
        z1 = self.fc1(x)
        a1 = F.relu(z1)
        z2 = self.fc2(a1)
        a2 = F.softmax(z2, dim=1)
        return {
            'input': a0.detach().numpy().flatten(),
            'hidden1': a1.detach().numpy().flatten(),
            'output_logits': z2.detach().numpy().flatten(),
            'output_probs': a2.detach().numpy().flatten(),
        }


class MNIST_MLP(nn.Module):
    """Dual hidden layer: 784 → 16 → 16 → 10 (compact interpretability-first model)."""
    def __init__(self, hidden1=16, hidden2=16):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.architecture = f"784 → {hidden1} → {hidden2} → 10"
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_layer_activations(self, x):
        """
        Returns activations at every layer for a single input.
        
        Returns:
            dict with keys: 'input', 'hidden1', 'hidden2', 'output_logits', 'output_probs'
        """
        with torch.no_grad():
            x = x.view(-1, 784).float()
            a0 = x.clone()
            z1 = self.fc1(x)
            a1 = F.relu(z1)
            z2 = self.fc2(a1)
            a2 = F.relu(z2)
            z3 = self.fc3(a2)
            a3 = F.softmax(z3, dim=1)
            return {
                'input': a0.numpy().flatten(),
                'hidden1': a1.numpy().flatten(),
                'hidden2': a2.numpy().flatten(),
                'output_logits': z3.numpy().flatten(),
                'output_probs': a3.numpy().flatten(),
            }

    def get_feature_maps(self, layer_idx=0):
        """
        Returns weight matrices reshaped as 28×28 heatmaps for the first layer,
        or raw weight matrices for deeper layers.
        
        Args:
            layer_idx: 0 for fc1 (input→hidden1), 1 for fc2 (hidden1→hidden2)
            
        Returns:
            numpy array of shape (n_neurons, 28, 28) for layer 0,
            or (n_neurons, hidden1) for layer 1
        """
        with torch.no_grad():
            if layer_idx == 0:
                weights = self.fc1.weight.numpy()  # (hidden1, 784)
                return weights.reshape(-1, 28, 28)
            elif layer_idx == 1:
                return self.fc2.weight.numpy()  # (hidden2, hidden1)
            else:
                return self.fc3.weight.numpy()  # (10, hidden2)

    def get_top_activating_neurons(self, x, layer_idx=0, k=10):
        """
        Returns the indices and values of the k most active neurons for input x.
        
        Args:
            x: input tensor (1, 1, 28, 28) or (1, 784)
            layer_idx: 0 for hidden1, 1 for hidden2
            k: number of top neurons to return
            
        Returns:
            indices (array), values (array)
        """
        activations = self.get_layer_activations(x)
        if layer_idx == 0:
            acts = activations['hidden1']
        elif layer_idx == 1:
            acts = activations['hidden2']
        else:
            acts = activations['output_probs']
        
        top_k_idx = np.argsort(acts)[-k:][::-1]
        return top_k_idx, acts[top_k_idx]

    @classmethod
    def load_pretrained(cls, path, hidden1=16, hidden2=16):
        """Load a pretrained model from a saved state dict."""
        model = cls(hidden1=hidden1, hidden2=hidden2)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        return model
    
    def count_parameters(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
