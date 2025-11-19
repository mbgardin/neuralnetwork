"""
Training script for NumPy neural network implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD, SGDMomentum, Adam
from src.utils import load_mnist, create_mini_batches, History


def plot_training_history(history, save_path='plots/numpy_training_history.png'):
    """
    Plot training and validation metrics.

    Parameters
    ----------
    history : dict
        Training history dictionary
    save_path : str
        Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs. Epoch', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-',
             label='Training Accuracy', linewidth=2)
    if history['val_acc']:
        ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-',
                 label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs. Epoch', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=True):
    """
    Train the neural network.

    Parameters
    ----------
    model : NeuralNetwork
        Neural network model
    X_train : ndarray
        Training data
    y_train : ndarray
        Training labels
    X_val : ndarray
        Validation data
    y_val : ndarray
        Validation labels
    epochs : int, default=10
        Number of training epochs
    batch_size : int, default=32
        Mini-batch size
    verbose : bool, default=True
        Whether to print progress

    Returns
    -------
    history : History
        Training history object
    """
    history = History()

    n_batches = len(X_train) // batch_size

    print("\nStarting training...")
    print(f"Total batches per epoch: {n_batches}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        batch_losses = []

        for batch_idx, (X_batch, y_batch) in enumerate(
            create_mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True)
        ):
            loss = model.train_step(X_batch, y_batch)
            batch_losses.append(loss)

            if verbose and (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(batch_losses[-100:])
                print(f"  Batch {batch_idx + 1}/{n_batches} - Loss: {avg_loss:.4f}", end='\r')

        train_loss = np.mean(batch_losses)
        train_loss_eval, train_acc = model.evaluate(X_train, y_train)

        val_loss, val_acc = model.evaluate(X_val, y_val)

        history.update(train_loss, train_acc, val_loss, val_acc)

        epoch_time = time.time() - epoch_start_time

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.1f}s - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    print("=" * 70)
    print("Training complete!")

    return history


def main():
    """
    Main training function.
    """
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("Neural Network From Scratch - NumPy Implementation")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_mnist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"\nDataset split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    model = NeuralNetwork()
    model.add(Dense(784, 128, weight_init='he'))
    model.add(ReLU())
    model.add(Dense(128, 64, weight_init='he'))
    model.add(ReLU())
    model.add(Dense(64, 10, weight_init='he'))
    model.add(Softmax())

    print("\n" + "=" * 70)
    model.summary()
    print()

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        loss=CrossEntropyLoss(),
        optimizer=optimizer
    )

    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {optimizer.learning_rate}")
    print()

    history = train(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=15,
        batch_size=128,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print("=" * 70)

    plot_training_history(history.get_history())

    print("\nDone!")


if __name__ == '__main__':
    main()
