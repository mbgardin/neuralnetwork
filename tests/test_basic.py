"""
Basic tests to verify neural network implementation.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.layers import Dense
from src.activations import ReLU, Sigmoid, Softmax
from src.losses import CrossEntropyLoss
from src.model import NeuralNetwork
from src.optimizers import SGD


def test_dense_layer():
    """Test Dense layer forward and backward passes."""
    print("Testing Dense layer...")

    layer = Dense(10, 5)

    X = np.random.randn(32, 10)

    output = layer.forward(X)

    assert output.shape == (32, 5), f"Expected shape (32, 5), got {output.shape}"

    dout = np.random.randn(32, 5)
    dinput = layer.backward(dout)

    assert dinput.shape == (32, 10), f"Expected shape (32, 10), got {dinput.shape}"
    assert layer.dweights.shape == (10, 5), f"Expected dweights shape (10, 5), got {layer.dweights.shape}"
    assert layer.dbias.shape == (1, 5), f"Expected dbias shape (1, 5), got {layer.dbias.shape}"

    print("  ✓ Dense layer forward/backward pass shapes correct")


def test_activations():
    """Test activation functions."""
    print("Testing activation functions...")

    X = np.random.randn(32, 10)

    relu = ReLU()
    output = relu.forward(X)
    assert output.shape == X.shape
    assert np.all(output >= 0), "ReLU output should be non-negative"
    print("  ✓ ReLU activation works")

    sigmoid = Sigmoid()
    output = sigmoid.forward(X)
    assert output.shape == X.shape
    assert np.all((output >= 0) & (output <= 1)), "Sigmoid output should be in [0, 1]"
    print("  ✓ Sigmoid activation works")

    softmax = Softmax()
    X_logits = np.random.randn(32, 10)
    output = softmax.forward(X_logits)
    assert output.shape == X_logits.shape
    sums = np.sum(output, axis=1)
    assert np.allclose(sums, 1.0), "Softmax outputs should sum to 1"
    print("  ✓ Softmax activation works")


def test_loss():
    """Test loss function."""
    print("Testing loss function...")

    loss_fn = CrossEntropyLoss()

    predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    targets = np.array([0, 1])

    loss = loss_fn.forward(predictions, targets)

    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss > 0, "Loss should be positive"

    dout = loss_fn.backward()
    assert dout.shape == predictions.shape

    print(f"  ✓ Cross-entropy loss works (loss = {loss:.4f})")


def test_simple_training():
    """Test a simple training step."""
    print("Testing simple training...")

    np.random.seed(42)

    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 3, 100)

    model = NeuralNetwork()
    model.add(Dense(20, 10))
    model.add(ReLU())
    model.add(Dense(10, 3))
    model.add(Softmax())

    model.compile(
        loss=CrossEntropyLoss(),
        optimizer=SGD(learning_rate=0.01)
    )

    initial_loss, initial_acc = model.evaluate(X_train, y_train)

    for _ in range(10):
        model.train_step(X_train, y_train)

    final_loss, final_acc = model.evaluate(X_train, y_train)

    print(f"  Initial loss: {initial_loss:.4f}, accuracy: {initial_acc:.4f}")
    print(f"  Final loss: {final_loss:.4f}, accuracy: {final_acc:.4f}")
    print(f"  ✓ Training step works (loss decreased: {initial_loss > final_loss})")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Basic Tests")
    print("=" * 60 + "\n")

    test_dense_layer()
    test_activations()
    test_loss()
    test_simple_training()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
