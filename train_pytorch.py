"""
Training script for PyTorch implementation.

This mirrors train_numpy.py for direct comparison.
"""

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os

from pytorch_model import MNISTNet, train_pytorch_model, evaluate_pytorch_model, prepare_dataloaders

try:
    from src.utils import load_mnist
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy utilities not available. Using PyTorch-only data loading.")


def plot_pytorch_history(history, save_path='plots/pytorch_training_history.png'):
    """
    Plot PyTorch training history.

    Parameters
    ----------
    history : dict
        Training history
    save_path : str
        Path to save plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs. Epoch (PyTorch)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-',
             label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-',
             label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs. Epoch (PyTorch)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """
    Main training function for PyTorch.
    """
    print("\n" + "=" * 70)
    print("Neural Network - PyTorch Implementation")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not NUMPY_AVAILABLE:
        print("\nError: NumPy utilities not available.")
        print("Please ensure NumPy, scikit-learn are installed:")
        print("  pip install numpy scikit-learn matplotlib")
        return

    print("\nLoading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"\nDataset split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    print("\nPreparing PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )

    print("\nCreating PyTorch model...")
    model = MNISTNet()

    print("\n" + "=" * 70)
    model.summary()
    print()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Optimizer: Adam")
    print(f"Learning rate: 0.001")
    print(f"Batch size: 128")
    print()

    start_time = time.time()

    history = train_pytorch_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=15,
        verbose=True
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    test_loss, test_acc = evaluate_pytorch_model(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Total training time: {total_time:.1f}s")
    print("=" * 70)

    plot_pytorch_history(history)

    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/pytorch_mnist.pth')
    print("Model saved to models/pytorch_mnist.pth")

    print("\nDone!")


if __name__ == '__main__':
    main()
