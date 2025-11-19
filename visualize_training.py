"""
Visualize what happens during training step-by-step.

This script shows how the network's predictions improve over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import Adam


def generate_simple_data(n_samples=200):
    """
    Generate a simple 2-class dataset (two clusters).

    Returns
    -------
    X : ndarray
        Features
    y : ndarray
        Labels
    """
    np.random.seed(42)

    X_class0 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X_class1 = np.random.randn(n_samples, 2) + np.array([-2, -2])

    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples, dtype=int),
                   np.ones(n_samples, dtype=int)])

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def plot_predictions(model, X, y, epoch, save_path):
    """
    Plot the model's predictions at a given epoch.

    Parameters
    ----------
    model : NeuralNetwork
        Neural network model
    X : ndarray
        Features
    y : ndarray
        True labels
    epoch : int
        Current epoch number
    save_path : str
        Path to save the plot
    """
    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_classes == y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)

    correct = pred_classes == y
    plt.scatter(X[correct, 0], X[correct, 1], c=y[correct],
               cmap='RdYlBu', edgecolors='black', s=50, marker='o',
               label='Correct', alpha=0.8)
    plt.scatter(X[~correct, 0], X[~correct, 1], c=y[~correct],
               cmap='RdYlBu', edgecolors='red', s=50, marker='X',
               label='Incorrect', alpha=0.8, linewidths=2)

    plt.colorbar(label='P(Class 1)')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(f'Epoch {epoch} - Accuracy: {accuracy*100:.1f}%',
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """
    Visualize training progression.
    """
    print("\n" + "=" * 70)
    print("VISUALIZING TRAINING PROGRESSION")
    print("=" * 70)

    np.random.seed(42)

    print("\nGenerating simple 2-class dataset...")
    X, y = generate_simple_data(n_samples=200)

    print(f"Dataset: {len(X)} samples, 2 classes")

    model = NeuralNetwork()
    model.add(Dense(2, 8))
    model.add(ReLU())
    model.add(Dense(8, 2))
    model.add(Softmax())

    model.compile(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.01))

    print("\nModel:")
    model.summary()

    epochs_to_save = [0, 1, 5, 10, 20, 50]

    os.makedirs('plots/progression', exist_ok=True)

    print("\nTraining and saving snapshots...")

    loss_history = []
    acc_history = []

    for epoch in range(51):
        loss = model.train_step(X, y)
        _, acc = model.evaluate(X, y)

        loss_history.append(loss)
        acc_history.append(acc)

        if epoch in epochs_to_save:
            save_path = f'plots/progression/epoch_{epoch:03d}.png'
            plot_predictions(model, X, y, epoch, save_path)
            print(f"  Saved snapshot for epoch {epoch} - Loss: {loss:.4f}, Acc: {acc:.4f}")

    final_loss, final_acc = model.evaluate(X, y)
    print(f"\nFinal accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

    print("\nCreating training curve...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, linewidth=2, color='steelblue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in acc_history], linewidth=2, color='coral')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/progression/training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved training curves to plots/progression/training_curves.png")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("\nCheck plots/progression/ folder for:")
    print("  - Snapshots at different epochs showing how decision boundary evolves")
    print("  - Training curves showing loss and accuracy over time")
    print("=" * 70)


if __name__ == '__main__':
    main()
