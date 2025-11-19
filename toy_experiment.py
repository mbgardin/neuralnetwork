"""
Toy experiment using a simple synthetic dataset.

This runs much faster than MNIST and lets us verify the implementation works.
We'll create a simple 2D classification problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD, Adam


def generate_spiral_data(n_samples=1000, n_classes=3, noise=0.1):
    """
    Generate spiral dataset for classification.

    Parameters
    ----------
    n_samples : int
        Number of samples per class
    n_classes : int
        Number of classes
    noise : float
        Standard deviation of noise

    Returns
    -------
    X : ndarray of shape (n_samples * n_classes, 2)
        Features
    y : ndarray of shape (n_samples * n_classes,)
        Labels
    """
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype=int)

    for class_idx in range(n_classes):
        idx = range(n_samples * class_idx, n_samples * (class_idx + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples) + \
            np.random.randn(n_samples) * noise

        X[idx] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[idx] = class_idx

    return X, y


def plot_decision_boundary(model, X, y, save_path='plots/toy_decision_boundary.png'):
    """
    Plot decision boundary of the model.

    Parameters
    ----------
    model : NeuralNetwork
        Trained model
    X : ndarray
        Features
    y : ndarray
        Labels
    save_path : str
        Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         edgecolors='black', s=50, alpha=0.8)
    plt.colorbar(scatter)

    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Decision Boundary - Spiral Classification', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Decision boundary plot saved to {save_path}")
    plt.close()


def main():
    """
    Run toy experiment with spiral dataset.
    """
    print("\n" + "=" * 70)
    print("TOY EXPERIMENT: Spiral Classification")
    print("=" * 70)

    np.random.seed(42)

    print("\nGenerating spiral dataset...")
    X_train, y_train = generate_spiral_data(n_samples=300, n_classes=3, noise=0.15)
    X_test, y_test = generate_spiral_data(n_samples=100, n_classes=3, noise=0.15)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: 2, Classes: 3")

    print("\n--- Experiment 1: Small Network with SGD ---")
    model1 = NeuralNetwork()
    model1.add(Dense(2, 16))
    model1.add(ReLU())
    model1.add(Dense(16, 3))
    model1.add(Softmax())
    model1.compile(loss=CrossEntropyLoss(), optimizer=SGD(learning_rate=0.1))

    print("\nTraining...")
    for epoch in range(100):
        loss = model1.train_step(X_train, y_train)

        if (epoch + 1) % 20 == 0:
            _, acc = model1.evaluate(X_train, y_train)
            print(f"Epoch {epoch+1}/100 - Loss: {loss:.4f} - Acc: {acc:.4f}")

    test_loss1, test_acc1 = model1.evaluate(X_test, y_test)
    print(f"\nTest Accuracy (SGD): {test_acc1:.4f} ({test_acc1*100:.2f}%)")

    print("\n--- Experiment 2: Deeper Network with Adam ---")
    model2 = NeuralNetwork()
    model2.add(Dense(2, 32))
    model2.add(ReLU())
    model2.add(Dense(32, 16))
    model2.add(ReLU())
    model2.add(Dense(16, 3))
    model2.add(Softmax())
    model2.compile(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.01))

    print("\nTraining...")
    for epoch in range(100):
        loss = model2.train_step(X_train, y_train)

        if (epoch + 1) % 20 == 0:
            _, acc = model2.evaluate(X_train, y_train)
            print(f"Epoch {epoch+1}/100 - Loss: {loss:.4f} - Acc: {acc:.4f}")

    test_loss2, test_acc2 = model2.evaluate(X_test, y_test)
    print(f"\nTest Accuracy (Adam): {test_acc2:.4f} ({test_acc2*100:.2f}%)")

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"SGD (small network):  {test_acc1*100:.2f}%")
    print(f"Adam (deep network):  {test_acc2*100:.2f}%")
    print("=" * 70)

    print("\nGenerating decision boundary visualization...")
    plot_decision_boundary(model2, X_test, y_test)

    print("\nVisualization tips:")
    print("- Check plots/toy_decision_boundary.png to see how the network")
    print("  learned to separate the spiral patterns")
    print("- Different colors show different predicted classes")
    print("- Points are the actual test samples")

    print("\n" + "=" * 70)
    print("Toy experiment complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
