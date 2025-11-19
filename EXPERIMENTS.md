# Experimentation Guide

This guide helps you experiment with the neural network implementation to understand how different choices affect performance.

## Available Experiment Scripts

### 1. Basic Tests (`tests/test_basic.py`)

**Purpose**: Verify the implementation works correctly

**Run**:
```bash
python tests/test_basic.py
```

**What it does**:
- Tests layer forward/backward passes
- Tests activation functions
- Tests loss computation
- Verifies training reduces loss

**Expected time**: < 5 seconds

---

### 2. Toy Experiment (`toy_experiment.py`)

**Purpose**: Quick visualization of learning on a simple 2D spiral dataset

**Run**:
```bash
python toy_experiment.py
```

**What it does**:
- Generates 3-class spiral dataset
- Trains two models (SGD vs Adam)
- Creates decision boundary visualization
- Shows which model learns better

**Expected time**: 10-20 seconds

**Output**: `plots/toy_decision_boundary.png`

**What to learn**:
- How different optimizers affect learning
- Visual understanding of decision boundaries
- Why deeper networks can capture more complex patterns

---

### 3. Training Visualization (`visualize_training.py`)

**Purpose**: See how the network learns step-by-step

**Run**:
```bash
python visualize_training.py
```

**What it does**:
- Creates snapshots at epochs 0, 1, 5, 10, 20, 50
- Shows how decision boundary evolves
- Plots training curves

**Expected time**: 15-30 seconds

**Output**:
- `plots/progression/epoch_*.png` - Snapshots at different stages
- `plots/progression/training_curves.png` - Loss and accuracy curves

**What to learn**:
- How the network starts with random predictions
- How the decision boundary gradually improves
- When the network converges

---

### 4. Full MNIST Training (`train_numpy.py`)

**Purpose**: Train on real MNIST dataset

**Run**:
```bash
python train_numpy.py
```

**What it does**:
- Downloads MNIST (70,000 images)
- Trains 3-layer network for 15 epochs
- Evaluates on test set
- Creates training curves

**Expected time**: 2-5 minutes (CPU)

**Expected accuracy**: 97-98%

**Output**: `plots/numpy_training_history.png`

---

### 5. Hyperparameter Comparison (`experiment.py`)

**Purpose**: Compare different configurations systematically

**Run**:
```bash
python experiment.py
```

**What it does**:
- Runs 5 different configurations on MNIST
- Compares: network size, optimizers, batch sizes
- Creates comparison charts

**Expected time**: 5-10 minutes

**Output**: `plots/experiment_comparison.png`

**What to learn**:
- Which optimizer works best
- How network depth affects performance
- Impact of batch size

---

## Quick Experiments You Can Try

### Experiment A: Learning Rate Impact

Edit `toy_experiment.py` and try different learning rates:

```python
# Line with optimizer
optimizer=Adam(learning_rate=0.001)  # Original
optimizer=Adam(learning_rate=0.1)    # Too high?
optimizer=Adam(learning_rate=0.00001) # Too low?
```

**Question**: What happens with very high or very low learning rates?

---

### Experiment B: Network Depth

Edit `toy_experiment.py` model architecture:

```python
# Very shallow (1 hidden layer)
model.add(Dense(2, 8))
model.add(ReLU())
model.add(Dense(8, 3))
model.add(Softmax())

# vs Deep (4 hidden layers)
model.add(Dense(2, 32))
model.add(ReLU())
model.add(Dense(32, 16))
model.add(ReLU())
model.add(Dense(16, 8))
model.add(ReLU())
model.add(Dense(8, 4))
model.add(ReLU())
model.add(Dense(4, 3))
model.add(Softmax())
```

**Question**: Does deeper always mean better?

---

### Experiment C: Activation Functions

Edit the activation in `toy_experiment.py`:

```python
model.add(ReLU())      # Original
model.add(Sigmoid())   # Try this
model.add(Tanh())      # Or this
```

**Question**: Which activation works best? Why?

---

### Experiment D: Optimizer Comparison

Run these three variations in `toy_experiment.py`:

```python
# Vanilla SGD
optimizer=SGD(learning_rate=0.1)

# SGD with Momentum
optimizer=SGDMomentum(learning_rate=0.1, momentum=0.9)

# Adam
optimizer=Adam(learning_rate=0.01)
```

**Question**: Which converges fastest? Which reaches highest accuracy?

---

### Experiment E: Batch Size Effects

In `train_numpy.py`, change batch size:

```python
history = train(
    model, X_train, y_train, X_val, y_val,
    epochs=5,
    batch_size=32,   # Small batches
    # batch_size=128,  # Medium (default)
    # batch_size=512,  # Large batches
)
```

**Question**: How does batch size affect training speed and accuracy?

---

## Understanding the Results

### Good Signs:
- Loss decreases steadily
- Training and validation accuracy both increase
- Decision boundaries look smooth and reasonable
- Test accuracy is close to validation accuracy

### Bad Signs:
- Loss stays flat or increases (learning rate too high/low?)
- Training accuracy high but validation low (overfitting)
- Loss oscillates wildly (learning rate too high?)
- Accuracy stuck at ~10% for MNIST (network not learning at all)

### Common Issues:

**Problem**: Loss is NaN
**Solution**: Lower learning rate, check for bugs in backward pass

**Problem**: Training is very slow
**Solution**: Increase batch size, use fewer epochs for quick tests

**Problem**: Accuracy stuck at random chance
**Solution**: Increase learning rate, train longer, try different optimizer

**Problem**: Overfitting (train >> test accuracy)
**Solution**: Currently not implemented, but you could add dropout or L2 regularization

---

## Suggested Exploration Path

1. **Start simple**: Run `tests/test_basic.py` to verify everything works
2. **Visualize learning**: Run `visualize_training.py` to see how networks learn
3. **Quick experiments**: Try `toy_experiment.py` with different settings
4. **Real dataset**: Run `train_numpy.py` to see performance on MNIST
5. **Compare systematically**: Run `experiment.py` to compare configurations

---

## Recording Your Results

Create a notebook to track experiments:

```markdown
## Experiment Log

### Experiment 1: Baseline
- Date: 2024-XX-XX
- Config: 2 layers, Adam, lr=0.001
- Result: 97.5% test accuracy
- Notes: Good baseline

### Experiment 2: Deeper Network
- Date: 2024-XX-XX
- Config: 4 layers, Adam, lr=0.001
- Result: 97.8% test accuracy
- Notes: Slight improvement, slower training
```

---

## Next Steps

After experimenting with these scripts:
1. Try creating your own dataset
2. Implement new layer types (e.g., Dropout)
3. Add regularization techniques
4. Compare with PyTorch implementation (Phase 5)
5. Try other datasets (Fashion-MNIST, CIFAR-10)

Happy experimenting!
