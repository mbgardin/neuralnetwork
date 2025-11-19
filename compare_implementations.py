"""
Compare NumPy and PyTorch implementations side-by-side.

Trains both models on the same data and compares:
- Training time
- Final accuracy
- Training curves
- Code complexity
"""

import time
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

try:
    import numpy as np
    import torch
    import torch.optim as optim
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Error: Required dependencies not available.")
    print("Please install: pip install numpy torch scikit-learn matplotlib")


def run_numpy_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=10):
    """
    Train NumPy implementation.

    Returns
    -------
    results : dict
        Training results
    """
    from src.model import NeuralNetwork
    from src.layers import Dense
    from src.activations import ReLU, Softmax
    from src.losses import CrossEntropyLoss
    from src.optimizers import Adam
    from src.utils import create_mini_batches, History

    print("\n" + "=" * 70)
    print("TRAINING NUMPY IMPLEMENTATION")
    print("=" * 70)

    model = NeuralNetwork()
    model.add(Dense(784, 128))
    model.add(ReLU())
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dense(64, 10))
    model.add(Softmax())

    model.compile(
        loss=CrossEntropyLoss(),
        optimizer=Adam(learning_rate=0.001)
    )

    print("\nNumPy Model:")
    model.summary()

    history = History()
    batch_size = 128

    start_time = time.time()

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        batch_losses = []

        for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size, shuffle=True):
            loss = model.train_step(X_batch, y_batch)
            batch_losses.append(loss)

        train_loss = np.mean(batch_losses)
        _, train_acc = model.evaluate(X_train, y_train)
        val_loss, val_acc = model.evaluate(X_val, y_val)

        history.update(train_loss, train_acc, val_loss, val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
              f"Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    training_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f"\nNumPy Results:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Training Time: {training_time:.1f}s")
    print("=" * 70)

    return {
        'name': 'NumPy',
        'test_loss': test_loss,
        'test_acc': test_acc,
        'training_time': training_time,
        'history': history.get_history()
    }


def run_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=10):
    """
    Train PyTorch implementation.

    Returns
    -------
    results : dict
        Training results
    """
    from pytorch_model import MNISTNet, train_pytorch_model, evaluate_pytorch_model, prepare_dataloaders

    print("\n" + "=" * 70)
    print("TRAINING PYTORCH IMPLEMENTATION")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    model = MNISTNet()

    print("\nPyTorch Model:")
    model.summary()

    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    history = train_pytorch_model(
        model, train_loader, val_loader, optimizer, device, epochs=epochs, verbose=True
    )

    training_time = time.time() - start_time

    test_loss, test_acc = evaluate_pytorch_model(model, test_loader, device)

    print(f"\nPyTorch Results:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Training Time: {training_time:.1f}s")
    print("=" * 70)

    return {
        'name': 'PyTorch',
        'test_loss': test_loss,
        'test_acc': test_acc,
        'training_time': training_time,
        'history': history
    }


def plot_comparison(numpy_results, pytorch_results, save_path='plots/comparison.png'):
    """
    Create side-by-side comparison plots.

    Parameters
    ----------
    numpy_results : dict
        NumPy training results
    pytorch_results : dict
        PyTorch training results
    save_path : str
        Path to save plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    numpy_hist = numpy_results['history']
    pytorch_hist = pytorch_results['history']
    epochs = range(1, len(numpy_hist['train_loss']) + 1)

    axes[0, 0].plot(epochs, numpy_hist['train_loss'], 'b-', label='NumPy', linewidth=2)
    axes[0, 0].plot(epochs, pytorch_hist['train_loss'], 'r-', label='PyTorch', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, numpy_hist['val_loss'], 'b-', label='NumPy', linewidth=2)
    axes[0, 1].plot(epochs, pytorch_hist['val_loss'], 'r-', label='PyTorch', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Comparison', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, [a*100 for a in numpy_hist['train_acc']], 'b-',
                    label='NumPy', linewidth=2)
    axes[1, 0].plot(epochs, [a*100 for a in pytorch_hist['train_acc']], 'r-',
                    label='PyTorch', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy Comparison', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, [a*100 for a in numpy_hist['val_acc']], 'b-',
                    label='NumPy', linewidth=2)
    axes[1, 1].plot(epochs, [a*100 for a in pytorch_hist['val_acc']], 'r-',
                    label='PyTorch', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy Comparison', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    implementations = ['NumPy', 'PyTorch']
    accuracies = [numpy_results['test_acc'] * 100, pytorch_results['test_acc'] * 100]
    times = [numpy_results['training_time'], pytorch_results['training_time']]

    ax1.bar(implementations, accuracies, color=['steelblue', 'coral'])
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Final Test Accuracy', fontweight='bold')
    ax1.set_ylim([min(accuracies) - 1, 100])
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(implementations, times, color=['steelblue', 'coral'])
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time', fontweight='bold')
    for i, v in enumerate(times):
        ax2.text(i, v + max(times)*0.02, f'{v:.1f}s', ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/comparison_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to plots/comparison_metrics.png")
    plt.close()


def print_final_comparison(numpy_results, pytorch_results):
    """Print final comparison table."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'NumPy':<20} {'PyTorch':<20}")
    print("-" * 70)

    print(f"{'Test Accuracy':<25} {numpy_results['test_acc']*100:>18.2f}% "
          f"{pytorch_results['test_acc']*100:>18.2f}%")

    print(f"{'Test Loss':<25} {numpy_results['test_loss']:>20.4f} "
          f"{pytorch_results['test_loss']:>20.4f}")

    print(f"{'Training Time':<25} {numpy_results['training_time']:>18.1f}s "
          f"{pytorch_results['training_time']:>18.1f}s")

    speedup = numpy_results['training_time'] / pytorch_results['training_time']
    print(f"{'Speedup (PyTorch)':<25} {'-':<20} {speedup:>18.2f}x")

    print("\n" + "=" * 70)

    print("\nKey Observations:")
    print("1. Both implementations achieve similar accuracy")
    print("2. PyTorch is typically faster due to optimization")
    print("3. NumPy implementation is more transparent/educational")
    print("4. PyTorch provides GPU acceleration support")
    print("=" * 70)


def main():
    """Run comparison."""
    if not DEPENDENCIES_AVAILABLE:
        return

    from src.utils import load_mnist

    print("\n" + "=" * 70)
    print("NUMPY vs PYTORCH COMPARISON")
    print("=" * 70)

    print("\nLoading MNIST dataset...")
    X_train, X_test, y_train, y_test = load_mnist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    epochs = 10
    print(f"\nBoth models will train for {epochs} epochs with identical settings:")
    print("  - Architecture: 784 → 128 → 64 → 10")
    print("  - Optimizer: Adam (lr=0.001)")
    print("  - Batch size: 128")

    numpy_results = run_numpy_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs)
    pytorch_results = run_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs)

    plot_comparison(numpy_results, pytorch_results)
    print_final_comparison(numpy_results, pytorch_results)

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
