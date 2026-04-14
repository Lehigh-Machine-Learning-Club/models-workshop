"""
Pre-computation engine for the Toy MLP training trajectory.

Runs the full training loop offline and serializes every checkpoint
(parameters, loss, accuracy, decision boundary grid) to a compressed
NumPy archive on disk. The Streamlit page then loads this archive and
plays it back via a slider — no live training during presentation.

Typical usage (from CLI):
    python scripts/precompute_toy_training.py

Typical usage (from Streamlit page):
    from src.precompute_toy import load_checkpoints
    checkpoints = load_checkpoints("models/toy_checkpoints.npz")
"""
import numpy as np
import os
from src.mlp import MLP, ACTIVATIONS, sigmoid


# ---- Default grid bounds (matches the moons dataset after normalization) ----
GRID_PAD = 0.1
GRID_RESOLUTION = 50  # Resolution for serialized decision boundary


def precompute_training(X, y, activation_name='Sigmoid', lr=0.5,
                        max_epochs=10000, checkpoint_every=5, seed=42,
                        grid_resolution=GRID_RESOLUTION,
                        early_stop_patience=200, convergence_threshold=1e-6):
    """
    Run the full training loop and capture checkpoint snapshots.

    Args:
        X: Training features, shape (N, 2)
        y: Training labels, shape (N, 1)
        activation_name: Name of hidden-layer activation function
        lr: Learning rate
        max_epochs: Maximum number of training epochs
        checkpoint_every: Save a checkpoint every N epochs
        seed: Random seed for weight initialization
        grid_resolution: Resolution for the decision boundary mesh
        early_stop_patience: Stop if no improvement for this many epochs
        convergence_threshold: Minimum loss improvement to count as progress

    Returns:
        dict with keys:
            'epochs':       1D array of epoch numbers
            'W1':           array of shape (n_ckpt, 2, 3)
            'b1':           array of shape (n_ckpt, 1, 3)
            'W2':           array of shape (n_ckpt, 3, 1)
            'b2':           array of shape (n_ckpt, 1, 1)
            'losses':       1D array of loss values
            'accuracies':   1D array of accuracy values
            'boundaries':   array of shape (n_ckpt, grid_res, grid_res)
            'grid_x':       1D linspace for x-axis
            'grid_y':       1D linspace for y-axis
            'activation':   string name
            'lr':           float
            'seed':         int
            'X':            training data (N, 2)
            'y':            training labels (N, 1)
    """
    mlp = MLP(seed=seed)
    act_fn, _ = ACTIVATIONS[activation_name]

    # Build the mesh grid once
    x_min, x_max = X[:, 0].min() - GRID_PAD, X[:, 0].max() + GRID_PAD
    y_min, y_max = X[:, 1].min() - GRID_PAD, X[:, 1].max() + GRID_PAD
    grid_x = np.linspace(x_min, x_max, grid_resolution)
    grid_y = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Storage lists
    ckpt_epochs = []
    ckpt_W1, ckpt_b1, ckpt_W2, ckpt_b2 = [], [], [], []
    ckpt_losses, ckpt_accs = [], []
    ckpt_boundaries = []

    # -- Always capture epoch 0 (initial random state) --
    mlp.forward(X, activation_name)
    loss0, acc0 = mlp.compute_loss_accuracy(y)
    Z0 = mlp.forward(grid_points, activation_name).reshape(xx.shape)
    mlp.forward(X, activation_name)  # restore cache

    ckpt_epochs.append(0)
    ckpt_W1.append(mlp.W1.copy())
    ckpt_b1.append(mlp.b1.copy())
    ckpt_W2.append(mlp.W2.copy())
    ckpt_b2.append(mlp.b2.copy())
    ckpt_losses.append(loss0)
    ckpt_accs.append(acc0)
    ckpt_boundaries.append(Z0)

    best_loss = loss0
    epochs_since_improvement = 0

    for epoch in range(1, max_epochs + 1):
        loss, acc = mlp.train_step(X, y, activation_name, lr)

        # Checkpoint?
        if epoch % checkpoint_every == 0 or epoch == max_epochs:
            Z = mlp.forward(grid_points, activation_name).reshape(xx.shape)
            mlp.forward(X, activation_name)  # restore cache for next train_step

            ckpt_epochs.append(epoch)
            ckpt_W1.append(mlp.W1.copy())
            ckpt_b1.append(mlp.b1.copy())
            ckpt_W2.append(mlp.W2.copy())
            ckpt_b2.append(mlp.b2.copy())
            ckpt_losses.append(loss)
            ckpt_accs.append(acc)
            ckpt_boundaries.append(Z)

        # Early stopping
        if loss < best_loss - convergence_threshold:
            best_loss = loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stop_patience and acc >= 0.99:
            # Capture final state if not already captured
            if ckpt_epochs[-1] != epoch:
                Z = mlp.forward(grid_points, activation_name).reshape(xx.shape)
                mlp.forward(X, activation_name)
                ckpt_epochs.append(epoch)
                ckpt_W1.append(mlp.W1.copy())
                ckpt_b1.append(mlp.b1.copy())
                ckpt_W2.append(mlp.W2.copy())
                ckpt_b2.append(mlp.b2.copy())
                ckpt_losses.append(loss)
                ckpt_accs.append(acc)
                ckpt_boundaries.append(Z)
            break

        if acc >= 1.0:
            if ckpt_epochs[-1] != epoch:
                Z = mlp.forward(grid_points, activation_name).reshape(xx.shape)
                mlp.forward(X, activation_name)
                ckpt_epochs.append(epoch)
                ckpt_W1.append(mlp.W1.copy())
                ckpt_b1.append(mlp.b1.copy())
                ckpt_W2.append(mlp.W2.copy())
                ckpt_b2.append(mlp.b2.copy())
                ckpt_losses.append(loss)
                ckpt_accs.append(acc)
                ckpt_boundaries.append(Z)
            break

    return {
        'epochs': np.array(ckpt_epochs),
        'W1': np.array(ckpt_W1),
        'b1': np.array(ckpt_b1),
        'W2': np.array(ckpt_W2),
        'b2': np.array(ckpt_b2),
        'losses': np.array(ckpt_losses),
        'accuracies': np.array(ckpt_accs),
        'boundaries': np.array(ckpt_boundaries),
        'grid_x': grid_x,
        'grid_y': grid_y,
        'activation': activation_name,
        'lr': lr,
        'seed': seed,
        'X': X,
        'y': y,
    }


def save_checkpoints(checkpoint_data, path):
    """Save pre-computed checkpoints to a compressed .npz file on disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        epochs=checkpoint_data['epochs'],
        W1=checkpoint_data['W1'],
        b1=checkpoint_data['b1'],
        W2=checkpoint_data['W2'],
        b2=checkpoint_data['b2'],
        losses=checkpoint_data['losses'],
        accuracies=checkpoint_data['accuracies'],
        boundaries=checkpoint_data['boundaries'],
        grid_x=checkpoint_data['grid_x'],
        grid_y=checkpoint_data['grid_y'],
        activation=np.array(checkpoint_data['activation']),
        lr=np.array(checkpoint_data['lr']),
        seed=np.array(checkpoint_data['seed']),
        X=checkpoint_data['X'],
        y=checkpoint_data['y'],
    )


def load_checkpoints(path):
    """
    Load pre-computed checkpoints from disk.

    Returns:
        dict with the same structure as precompute_training() output
    """
    data = np.load(path, allow_pickle=False)
    return {
        'epochs': data['epochs'],
        'W1': data['W1'],
        'b1': data['b1'],
        'W2': data['W2'],
        'b2': data['b2'],
        'losses': data['losses'],
        'accuracies': data['accuracies'],
        'boundaries': data['boundaries'],
        'grid_x': data['grid_x'],
        'grid_y': data['grid_y'],
        'activation': str(data['activation']),
        'lr': float(data['lr']),
        'seed': int(data['seed']),
        'X': data['X'],
        'y': data['y'],
    }


def restore_mlp_from_checkpoint(checkpoint_data, idx):
    """
    Create an MLP instance with parameters restored from checkpoint index.

    Args:
        checkpoint_data: dict from load_checkpoints()
        idx: checkpoint index

    Returns:
        MLP instance with restored parameters
    """
    mlp = MLP(seed=0)  # seed doesn't matter, we overwrite params
    mlp.W1 = checkpoint_data['W1'][idx].copy()
    mlp.b1 = checkpoint_data['b1'][idx].copy()
    mlp.W2 = checkpoint_data['W2'][idx].copy()
    mlp.b2 = checkpoint_data['b2'][idx].copy()
    return mlp
