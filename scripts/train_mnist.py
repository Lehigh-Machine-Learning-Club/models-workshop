"""
MNIST Training & Architecture Comparison Script.

 Trains both architectures:
  - Baseline: 784 → 64 → 10  (1 hidden layer)
  - Compact Deep: 784 → 16 → 16 → 10  (2 hidden layers, interpretability-first)

Compares accuracy, training time, and parameter count.
Saves the winner's weights + training history.

Usage:
    python scripts/train_mnist.py
    python scripts/train_mnist.py --epochs 2 --quick-test
"""
import sys
import os
import json
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from src.mnist_mlp import MNIST_MLP, MNIST_MLP_Small


def train_model(model, train_loader, test_loader, epochs=15, lr=0.001, device='cpu',
                l1_lambda=0.0, l1_target='none'):
    """Train a model and return training history."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_times': [],
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            ce_loss = criterion(outputs, batch_y)
            l1_penalty = torch.tensor(0.0, device=device)
            if l1_lambda > 0:
                if l1_target == 'fc1':
                    l1_penalty = model.fc1.weight.abs().mean()
                elif l1_target == 'all':
                    penalties = [p.abs().mean() for p in model.parameters() if p.requires_grad and p.ndim >= 2]
                    if penalties:
                        l1_penalty = torch.stack(penalties).mean()
            loss = ce_loss + (l1_lambda * l1_penalty)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        epoch_time = time.time() - start
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_loss /= test_total
        test_acc = test_correct / test_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    return history


def get_predictions(model, test_loader, device='cpu'):
    """Get all predictions and true labels for the test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description='Train MNIST MLP models')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with 2 epochs')
    parser.add_argument('--l1-lambda', type=float, default=0.0,
                        help='Optional L1 regularization strength (default: 0.0 = disabled)')
    parser.add_argument('--l1-target', type=str, default='none', choices=['none', 'fc1', 'all'],
                        help='Where to apply L1 regularization when enabled')
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 2
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Data setup
    print("="*60)
    print("MNIST MLP Training & Architecture Comparison")
    print("="*60)
    print(f"\nDownloading/loading MNIST dataset...")
    print(f"L1 regularization: lambda={args.l1_lambda} target={args.l1_target}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download to a data directory inside the project
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ---- Train Baseline Model ----
    print(f"\n{'='*60}")
    print("Model A: MNIST_MLP_Small (784 → 64 → 10)")
    model_small = MNIST_MLP_Small(hidden_size=64)
    n_params_small = sum(p.numel() for p in model_small.parameters())
    print(f"Parameters: {n_params_small:,}")
    print(f"{'='*60}")
    
    history_small = train_model(model_small, train_loader, test_loader, 
                                 epochs=args.epochs, lr=args.lr, device=device,
                                 l1_lambda=args.l1_lambda, l1_target=args.l1_target)
    
    # ---- Train Compact-Deep Model ----
    print(f"\n{'='*60}")
    print("Model B: MNIST_MLP (784 → 16 → 16 → 10)")
    model_large = MNIST_MLP(hidden1=16, hidden2=16)
    n_params_large = sum(p.numel() for p in model_large.parameters())
    print(f"Parameters: {n_params_large:,}")
    print(f"{'='*60}")
    
    history_large = train_model(model_large, train_loader, test_loader,
                                 epochs=args.epochs, lr=args.lr, device=device,
                                 l1_lambda=args.l1_lambda, l1_target=args.l1_target)
    
    # ---- Comparison ----
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    
    comparison = {
        'small': {
            'architecture': '784 → 64 → 10',
            'regularization': {'l1_lambda': args.l1_lambda, 'l1_target': args.l1_target},
            'parameters': n_params_small,
            'final_test_acc': history_small['test_acc'][-1],
            'final_test_loss': history_small['test_loss'][-1],
            'avg_epoch_time': np.mean(history_small['epoch_times']),
            'total_time': sum(history_small['epoch_times']),
        },
        'compact': {
            'architecture': '784 → 16 → 16 → 10',
            'regularization': {'l1_lambda': args.l1_lambda, 'l1_target': args.l1_target},
            'parameters': n_params_large,
            'final_test_acc': history_large['test_acc'][-1],
            'final_test_loss': history_large['test_loss'][-1],
            'avg_epoch_time': np.mean(history_large['epoch_times']),
            'total_time': sum(history_large['epoch_times']),
        }
    }
    
    print(f"{'Metric':<25} {'Baseline (784→64→10)':<25} {'Compact (784→16→16→10)':<25}")
    print("-"*75)
    print(f"{'Parameters':<25} {n_params_small:<25,} {n_params_large:<25,}")
    print(f"{'Test Accuracy':<25} {history_small['test_acc'][-1]*100:<25.2f} {history_large['test_acc'][-1]*100:<25.2f}")
    print(f"{'Test Loss':<25} {history_small['test_loss'][-1]:<25.4f} {history_large['test_loss'][-1]:<25.4f}")
    print(f"{'Avg Epoch Time (s)':<25} {np.mean(history_small['epoch_times']):<25.2f} {np.mean(history_large['epoch_times']):<25.2f}")
    
    # Save the primary model (compact deep)
    # Move model to CPU for saving
    model_large = model_large.cpu()
    model_large.eval()
    model_path = os.path.join(models_dir, 'mnist_mlp.pt')
    torch.save(model_large.state_dict(), model_path)
    print(f"\n✓ Compact model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(models_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'small': history_small,
            'compact': history_large,
        }, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Save comparison
    comparison_path = os.path.join(models_dir, 'architecture_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"✓ Comparison saved to {comparison_path}")
    
    # Get and save predictions for confusion matrix
    preds, labels, probs = get_predictions(model_large, test_loader, device='cpu')
    np.savez(
        os.path.join(models_dir, 'test_predictions.npz'),
        predictions=preds, labels=labels, probabilities=probs
    )
    print(f"✓ Test predictions saved")
    
    # Save some sample images for the data explorer
    sample_images = []
    sample_labels = []
    for digit in range(10):
        mask = test_dataset.targets == digit
        indices = mask.nonzero(as_tuple=True)[0][:5]
        for idx in indices:
            img, lbl = test_dataset[idx]
            sample_images.append(img.numpy())
            sample_labels.append(lbl)
    
    np.savez(
        os.path.join(models_dir, 'sample_digits.npz'),
        images=np.array(sample_images),
        labels=np.array(sample_labels)
    )
    print(f"✓ Sample digits saved")
    
    # Also save some misclassified examples
    wrong_mask = preds != labels
    wrong_indices = np.where(wrong_mask)[0][:20]  # First 20 errors
    wrong_images = []
    for idx in wrong_indices:
        img, _ = test_dataset[idx]
        wrong_images.append(img.numpy())
    
    np.savez(
        os.path.join(models_dir, 'misclassified.npz'),
        images=np.array(wrong_images) if len(wrong_images) > 0 else np.array([]),
        true_labels=labels[wrong_indices],
        pred_labels=preds[wrong_indices],
        probabilities=probs[wrong_indices],
    )
    print(f"✓ Misclassified examples saved ({len(wrong_indices)} samples)")
    
    print(f"\n{'='*60}")
    print(f"DONE! Primary model: Compact 784→16→16→10 with {history_large['test_acc'][-1]*100:.2f}% test accuracy")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
