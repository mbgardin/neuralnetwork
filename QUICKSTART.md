# Quick Start Guide

Get started with the neural network implementation in under 5 minutes!

## Prerequisites

```bash
pip install numpy scikit-learn matplotlib
```

## 1. Verify Installation (5 seconds)

```bash
python tests/test_basic.py
```

**Expected output**: All tests pass ✓

---

## 2. See Learning in Action (20 seconds)

```bash
python toy_experiment.py
```

**What happens**:
- Trains on 3-class spiral dataset
- Compares SGD vs Adam optimizer
- Creates visualization: `plots/toy_decision_boundary.png`

**Expected accuracy**: 85-95%

---

## 3. Visualize Training Process (30 seconds)

```bash
python visualize_training.py
```

**What happens**:
- Shows snapshots at epochs 0, 1, 5, 10, 20, 50
- Creates decision boundary evolution
- Saves to: `plots/progression/`

**What to notice**:
- Random predictions at epoch 0
- Gradual improvement in decision boundary
- Convergence around epoch 20-30

---

## 4. Train on Real Data (3-5 minutes)

```bash
python train_numpy.py
```

**What happens**:
- Downloads MNIST (first run only)
- Trains 3-layer network for 15 epochs
- Evaluates on 10,000 test images

**Expected accuracy**: 97-98%

**Output**: `plots/numpy_training_history.png`

---

## 5. Compare Configurations (8-10 minutes)

```bash
python experiment.py
```

**What happens**:
- Runs 5 different configurations
- Compares network sizes and optimizers
- Creates comparison chart

**Output**: `plots/experiment_comparison.png`

---

## Understanding the Code

### Build a Model

```python
from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import Adam

# Define architecture
model = NeuralNetwork()
model.add(Dense(784, 128))  # Input → Hidden
model.add(ReLU())
model.add(Dense(128, 10))   # Hidden → Output
model.add(Softmax())

# Compile
model.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.001)
)

# See architecture
model.summary()
```

### Train

```python
# Single training step
loss = model.train_step(X_batch, y_batch)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
```

---

## File Structure

```
project/
├── src/                    # Core implementation
│   ├── layers.py          # Dense layer
│   ├── activations.py     # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py          # CrossEntropy, MSE
│   ├── model.py           # NeuralNetwork class
│   ├── optimizers.py      # SGD, Momentum, Adam
│   └── utils.py           # Data loading, metrics
│
├── train_numpy.py         # Full MNIST training
├── toy_experiment.py      # Quick 2D visualization
├── visualize_training.py  # Training progression
├── experiment.py          # Compare configurations
│
├── tests/test_basic.py    # Verification tests
│
├── MATH_OVERVIEW.md       # Mathematical explanation
├── EXPERIMENTS.md         # Detailed experimentation guide
├── USAGE.md              # Comprehensive usage guide
└── QUICKSTART.md         # This file
```

---

## Quick Experiments

### Change Learning Rate

Edit any script:
```python
optimizer=Adam(learning_rate=0.001)  # Default
optimizer=Adam(learning_rate=0.01)   # 10x higher
optimizer=Adam(learning_rate=0.0001) # 10x lower
```

### Try Different Optimizer

```python
optimizer=SGD(learning_rate=0.01)
optimizer=SGDMomentum(learning_rate=0.01, momentum=0.9)
optimizer=Adam(learning_rate=0.001)
```

### Add More Layers

```python
model.add(Dense(784, 256))
model.add(ReLU())
model.add(Dense(256, 128))  # Extra layer
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())
```

---

## Next Steps

1. **Read the math**: `MATH_OVERVIEW.md` explains how it all works
2. **Experiment**: Try suggestions in `EXPERIMENTS.md`
3. **Go deeper**: Read full usage guide in `USAGE.md`
4. **Compare**: Implement PyTorch version (coming soon)

---

## Troubleshooting

**Q**: Script fails with "No module named 'numpy'"
**A**: Install dependencies: `pip install -r requirements.txt`

**Q**: Training is slow
**A**: Start with `toy_experiment.py` for quick testing

**Q**: Want to understand the code
**A**: Read `MATH_OVERVIEW.md` first, then explore `src/` files

**Q**: How do I know if it's working?
**A**: Run `tests/test_basic.py` - should pass all tests

---

**Need help?** Check `USAGE.md` for detailed documentation or `EXPERIMENTS.md` for experimentation ideas!
