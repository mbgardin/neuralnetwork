# Usage Guide

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Running the Code

### Quick Test

To verify the implementation works:

```bash
python tests/test_basic.py
```

This runs basic tests on:
- Dense layer forward/backward passes
- Activation functions
- Loss computation
- Simple training loop

### Train the NumPy Model

```bash
python train_numpy.py
```

This will:
1. Download and preprocess MNIST dataset
2. Build a 3-layer neural network (784 → 128 → 64 → 10)
3. Train for 15 epochs using Adam optimizer
4. Evaluate on test set
5. Save training curves to `plots/numpy_training_history.png`

**Expected output:**
- Test accuracy: ~97-98%
- Training time: ~2-5 minutes (CPU)

## Project Structure

```
src/
├── layers.py       # Dense layer implementation
├── activations.py  # ReLU, Sigmoid, Tanh, Softmax
├── losses.py       # Cross-entropy loss
├── model.py        # NeuralNetwork class
├── optimizers.py   # SGD, SGDMomentum, Adam
└── utils.py        # Data loading, mini-batches, metrics
```

## Example: Building a Custom Model

```python
from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import Adam

# Create model
model = NeuralNetwork()
model.add(Dense(784, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())

# Compile with loss and optimizer
model.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.001)
)

# Train
for epoch in range(10):
    for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size=128):
        loss = model.train_step(X_batch, y_batch)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

## Customization

### Change Architecture

Edit `train_numpy.py`:

```python
model.add(Dense(784, 256))  # Larger hidden layer
model.add(ReLU())
model.add(Dense(256, 128))  # Add another layer
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())
```

### Try Different Optimizers

```python
# Vanilla SGD
optimizer = SGD(learning_rate=0.01)

# SGD with momentum
optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)

# Adam (recommended)
optimizer = Adam(learning_rate=0.001)
```

### Adjust Hyperparameters

```python
history = train(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=20,           # More epochs
    batch_size=64,       # Smaller batches
    verbose=True
)
```

## Next Steps

1. **Understand the math**: Read `MATH_OVERVIEW.md`
2. **Run the tests**: `python tests/test_basic.py`
3. **Train the model**: `python train_numpy.py`
4. **Experiment**: Try different architectures and hyperparameters
5. **Compare with PyTorch**: (Coming in Phase 5)

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'numpy'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: Training is very slow
**Solution**:
- Reduce batch size
- Use fewer epochs
- Simplify architecture
- Consider using PyTorch for GPU acceleration (Phase 5)

**Issue**: Low accuracy
**Solution**:
- Train for more epochs
- Adjust learning rate
- Try different optimizer (Adam usually works best)
- Add more layers or neurons
