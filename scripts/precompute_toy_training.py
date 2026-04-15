"""
CLI script to pre-compute and serialize the Toy MLP training trajectory.

Generates checkpoint files for each activation function so the Streamlit
page can play back smooth animations without any live training.

Usage:
    python scripts/precompute_toy_training.py
    python scripts/precompute_toy_training.py --max-epochs 5000 --checkpoint-every 10
"""
import sys
import os
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import get_poisonous_fruit_data
from src.precompute_toy import precompute_training, save_checkpoints, DEFAULT_ACTIVATION_LRS


def main():
    parser = argparse.ArgumentParser(description='Pre-compute toy MLP training checkpoints')
    parser.add_argument('--max-epochs', type=int, default=2600,
                        help='Maximum training epochs per activation function')
    parser.add_argument('--checkpoint-every', type=int, default=2,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for weight initialization')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: models/)')
    parser.add_argument('--activations', nargs='+', default=None,
                        help='Which activation functions to compute (default: all)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(output_dir, exist_ok=True)

    X, y = get_poisonous_fruit_data()

    activations_to_run = args.activations or list(DEFAULT_ACTIVATION_LRS.keys())

    print("=" * 60)
    print("Toy MLP Training Pre-Computation")
    print("=" * 60)
    print(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Checkpoint every: {args.checkpoint_every} epochs")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print(f"  Activations: {activations_to_run}")
    print()

    for act_name in activations_to_run:
        if act_name not in DEFAULT_ACTIVATION_LRS:
            print(f"⚠️  Unknown activation '{act_name}', skipping.")
            continue

        lr = DEFAULT_ACTIVATION_LRS[act_name]
        print(f"{'─' * 60}")
        print(f"  Computing: {act_name}  (lr={lr})")

        start = time.time()
        checkpoints = precompute_training(
            X, y,
            activation_name=act_name,
            lr=lr,
            max_epochs=args.max_epochs,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
        )
        elapsed = time.time() - start

        n_ckpts = len(checkpoints['epochs'])
        final_epoch = int(checkpoints['epochs'][-1])
        final_loss = float(checkpoints['losses'][-1])
        final_acc = float(checkpoints['accuracies'][-1])

        # Save to disk
        safe_name = act_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        filename = f"toy_checkpoints_{safe_name}.npz"
        filepath = os.path.join(output_dir, filename)
        save_checkpoints(checkpoints, filepath)

        filesize_mb = os.path.getsize(filepath) / (1024 * 1024)

        print(f"  ✓ {n_ckpts} checkpoints  |  Final: epoch {final_epoch}, "
              f"loss={final_loss:.4f}, acc={final_acc*100:.1f}%")
        print(f"  ✓ Saved: {filepath}  ({filesize_mb:.2f} MB)")
        print(f"  ✓ Time: {elapsed:.2f}s")

    print(f"\n{'=' * 60}")
    print("Done! All checkpoint files saved.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
