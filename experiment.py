"""
Experimentation script for testing different neural network configurations.

This script allows quick testing of different:
- Architectures
- Optimizers
- Learning rates
- Batch sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Sigmoid, Tanh, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD, SGDMomentum, Adam
from src.utils import load_mnist, create_mini_batches, History


def quick_experiment(
    architecture,
    optimizer,
    epochs=5,
    batch_size=128,
    experiment_name="experiment"
):
    """
    Run a quick experiment with given configuration.

    Parameters
    ----------
    architecture : list of tuples
        List of (layer_size, activation) tuples
        Example: [(128, 'relu'), (64, 'relu'), (10, 'softmax')]
    optimizer : Optimizer
        Optimizer instance
    epochs : int
        Number of training epochs
    batch_size : int
        Mini-batch size
    experiment_name : str
        Name for this experiment

    Returns
    -------
    results : dict
        Dictionary with test accuracy, loss, and training time
    """
    print("\n" + "=" * 70)
    print(f"Experiment: {experiment_name}")
    print("=" * 70)

    np.random.seed(42)

    print("\nLoading data...")
    X_train_full, X_test, y_train_full, y_test = load_mnist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    model = NeuralNetwork()

    input_size = 784
    for i, (layer_size, activation) in enumerate(architecture):
        model.add(Dense(input_size, layer_size))

        if activation.lower() == 'relu':
            model.add(ReLU())
        elif activation.lower() == 'sigmoid':
            model.add(Sigmoid())
        elif activation.lower() == 'tanh':
            model.add(Tanh())
        elif activation.lower() == 'softmax':
            model.add(Softmax())

        input_size = layer_size

    print("\nModel architecture:")
    model.compile(loss=CrossEntropyLoss(), optimizer=optimizer)
    model.summary()

    print(f"\nOptimizer: {optimizer.__class__.__name__}")
    if hasattr(optimizer, 'learning_rate'):
        print(f"Learning rate: {optimizer.learning_rate}")
    if hasattr(optimizer, 'momentum'):
        print(f"Momentum: {optimizer.momentum}")

    start_time = time.time()
    history = History()

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

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("=" * 70)
    print(f"Results for {experiment_name}:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Training Time: {training_time:.1f}s")
    print("=" * 70)

    return {
        'name': experiment_name,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'training_time': training_time,
        'history': history.get_history()
    }


def compare_experiments(results_list):
    """
    Compare multiple experiment results.

    Parameters
    ----------
    results_list : list of dict
        List of result dictionaries from quick_experiment
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL EXPERIMENTS")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Test Acc':<12} {'Test Loss':<12} {'Time (s)':<10}")
    print("-" * 80)

    for result in results_list:
        print(f"{result['name']:<25} {result['test_acc']*100:>10.2f}% "
              f"{result['test_loss']:>10.4f}  {result['training_time']:>10.1f}")

    print("=" * 80)

    best_acc = max(results_list, key=lambda x: x['test_acc'])
    fastest = min(results_list, key=lambda x: x['training_time'])

    print(f"\nBest Accuracy: {best_acc['name']} ({best_acc['test_acc']*100:.2f}%)")
    print(f"Fastest: {fastest['name']} ({fastest['training_time']:.1f}s)")

    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [r['name'] for r in results_list]
    accuracies = [r['test_acc'] * 100 for r in results_list]
    times = [r['training_time'] for r in results_list]

    axes[0].bar(range(len(names)), accuracies, color='steelblue')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(names)), times, color='coral')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('Training Time Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/experiment_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to plots/experiment_comparison.png")
    plt.close()


def main():
    """
    Run multiple experiments to compare different configurations.
    """
    print("\n" + "=" * 80)
    print("NEURAL NETWORK EXPERIMENTATION")
    print("=" * 80)

    results = []

    print("\n[1/5] Baseline: Small network, SGD")
    results.append(quick_experiment(
        architecture=[(64, 'relu'), (10, 'softmax')],
        optimizer=SGD(learning_rate=0.01),
        epochs=5,
        batch_size=128,
        experiment_name="Small + SGD"
    ))

    print("\n[2/5] Baseline with momentum")
    results.append(quick_experiment(
        architecture=[(64, 'relu'), (10, 'softmax')],
        optimizer=SGDMomentum(learning_rate=0.01, momentum=0.9),
        epochs=5,
        batch_size=128,
        experiment_name="Small + Momentum"
    ))

    print("\n[3/5] Medium network with Adam")
    results.append(quick_experiment(
        architecture=[(128, 'relu'), (64, 'relu'), (10, 'softmax')],
        optimizer=Adam(learning_rate=0.001),
        epochs=5,
        batch_size=128,
        experiment_name="Medium + Adam"
    ))

    print("\n[4/5] Large network with Adam")
    results.append(quick_experiment(
        architecture=[(256, 'relu'), (128, 'relu'), (64, 'relu'), (10, 'softmax')],
        optimizer=Adam(learning_rate=0.001),
        epochs=5,
        batch_size=128,
        experiment_name="Large + Adam"
    ))

    print("\n[5/5] Small batch size experiment")
    results.append(quick_experiment(
        architecture=[(128, 'relu'), (64, 'relu'), (10, 'softmax')],
        optimizer=Adam(learning_rate=0.001),
        epochs=5,
        batch_size=32,
        experiment_name="Medium + Small Batch"
    ))

    compare_experiments(results)

    print("\n" + "=" * 80)
    print("All experiments complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
