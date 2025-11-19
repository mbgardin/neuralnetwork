"""
Minimal test of neural network implementation without external dependencies.
Uses only Python standard library + our implementation.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class SimpleArray:
    """Ultra-minimal numpy-like array for testing."""

    def __init__(self, data):
        if isinstance(data, list):
            self.data = data
            self.shape = (len(data), len(data[0]) if isinstance(data[0], list) else 1)
        else:
            self.data = data
            self.shape = ()

    def __repr__(self):
        return f"Array(shape={self.shape})"


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.layers import Dense
        print("  ✓ layers.py imported")
    except Exception as e:
        print(f"  ✗ layers.py failed: {e}")
        return False

    try:
        from src.activations import ReLU, Sigmoid, Tanh, Softmax
        print("  ✓ activations.py imported")
    except Exception as e:
        print(f"  ✗ activations.py failed: {e}")
        return False

    try:
        from src.losses import CrossEntropyLoss, MSELoss
        print("  ✓ losses.py imported")
    except Exception as e:
        print(f"  ✗ losses.py failed: {e}")
        return False

    try:
        from src.model import NeuralNetwork
        print("  ✓ model.py imported")
    except Exception as e:
        print(f"  ✗ model.py failed: {e}")
        return False

    try:
        from src.optimizers import SGD, SGDMomentum, Adam
        print("  ✓ optimizers.py imported")
    except Exception as e:
        print(f"  ✗ optimizers.py failed: {e}")
        return False

    try:
        from src.utils import compute_accuracy, History, EarlyStopping
        print("  ✓ utils.py imported (partial)")
    except Exception as e:
        print(f"  ✗ utils.py failed: {e}")
        return False

    return True


def test_code_structure():
    """Test that classes are properly structured."""
    print("\nTesting code structure...")

    from src.layers import Dense
    from src.activations import ReLU
    from src.model import NeuralNetwork
    from src.optimizers import Adam
    from src.losses import CrossEntropyLoss

    try:
        layer = Dense(10, 5)
        assert hasattr(layer, 'forward'), "Dense missing forward method"
        assert hasattr(layer, 'backward'), "Dense missing backward method"
        print("  ✓ Dense layer has forward/backward methods")
    except Exception as e:
        print(f"  ✗ Dense layer structure: {e}")
        return False

    try:
        activation = ReLU()
        assert hasattr(activation, 'forward'), "ReLU missing forward method"
        assert hasattr(activation, 'backward'), "ReLU missing backward method"
        print("  ✓ ReLU has forward/backward methods")
    except Exception as e:
        print(f"  ✗ ReLU structure: {e}")
        return False

    try:
        model = NeuralNetwork()
        assert hasattr(model, 'add'), "NeuralNetwork missing add method"
        assert hasattr(model, 'compile'), "NeuralNetwork missing compile method"
        assert hasattr(model, 'forward'), "NeuralNetwork missing forward method"
        assert hasattr(model, 'backward'), "NeuralNetwork missing backward method"
        assert hasattr(model, 'train_step'), "NeuralNetwork missing train_step method"
        print("  ✓ NeuralNetwork has all required methods")
    except Exception as e:
        print(f"  ✗ NeuralNetwork structure: {e}")
        return False

    try:
        optimizer = Adam(learning_rate=0.001)
        assert hasattr(optimizer, 'update'), "Adam missing update method"
        print("  ✓ Adam optimizer has update method")
    except Exception as e:
        print(f"  ✗ Adam structure: {e}")
        return False

    try:
        loss = CrossEntropyLoss()
        assert hasattr(loss, 'forward'), "Loss missing forward method"
        assert hasattr(loss, 'backward'), "Loss missing backward method"
        print("  ✓ CrossEntropyLoss has forward/backward methods")
    except Exception as e:
        print(f"  ✗ Loss structure: {e}")
        return False

    return True


def test_model_building():
    """Test that a model can be built."""
    print("\nTesting model building...")

    from src.model import NeuralNetwork
    from src.layers import Dense
    from src.activations import ReLU, Softmax
    from src.losses import CrossEntropyLoss
    from src.optimizers import Adam

    try:
        model = NeuralNetwork()
        model.add(Dense(784, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))
        model.add(Softmax())

        assert len(model.layers) == 4, f"Expected 4 layers, got {len(model.layers)}"
        print(f"  ✓ Model built successfully with {len(model.layers)} layers")

        model.compile(
            loss=CrossEntropyLoss(),
            optimizer=Adam(learning_rate=0.001)
        )

        assert model.loss_function is not None, "Loss function not set"
        assert model.optimizer is not None, "Optimizer not set"
        print("  ✓ Model compiled successfully")

        config = model.get_config()
        print(f"  ✓ Model config: {config['num_layers']} layers, "
              f"{config['total_params']:,} parameters")

        return True
    except Exception as e:
        print(f"  ✗ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MINIMAL IMPLEMENTATION TEST (No External Dependencies)")
    print("=" * 70 + "\n")

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_code_structure()
    all_passed &= test_model_building()

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nThe neural network implementation is structurally correct.")
        print("To run full tests with training, install dependencies:")
        print("  pip install numpy scikit-learn matplotlib")
        print("\nThen run:")
        print("  python tests/test_basic.py")
        print("  python toy_experiment.py")
        print("  python train_numpy.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("Check the error messages above.")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
