import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_MLP(nn.Module):
    def __init__(self, hidden_size=64):
        super(MNIST_MLP, self).__init__()
        # Flattened 28x28 image = 784
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        # Flatten image (Batch Size, 1, 28, 28) -> (Batch Size, 784)
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
