# NumPy vs PyTorch: Implementation Comparison

This document provides a detailed comparison between our NumPy implementation and the equivalent PyTorch implementation.

---

## Quick Summary

| Aspect | NumPy Implementation | PyTorch Implementation |
|--------|---------------------|----------------------|
| **Lines of Code** | ~1,200 lines (6 modules) | ~250 lines (2 files) |
| **Complexity** | Explicit everything | High-level API |
| **Speed** | Slower (CPU-only) | Faster (optimized, GPU support) |
| **Transparency** | Full control, see everything | Abstracted operations |
| **Use Case** | Learning, understanding | Production, research |
| **Dependencies** | NumPy only | PyTorch framework |

---

## Architecture Comparison

Both implementations use the identical architecture:
```
Input (784) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (10) → Softmax/Output
```

Total parameters: **109,386**

---

## Code Comparison

### Building a Model

**NumPy Implementation:**
```python
from src.model import NeuralNetwork
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import Adam

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
```

**PyTorch Implementation:**
```python
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MNISTNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

**Analysis:**
- NumPy: More explicit, shows layer-by-layer construction
- PyTorch: More compact, standard pattern
- Both are readable and maintainable

---

### Training Loop

**NumPy Implementation:**
```python
for epoch in range(epochs):
    for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size):
        # Forward pass
        predictions = model.forward(X_batch)

        # Compute loss
        loss = loss_fn.forward(predictions, y_batch)

        # Backward pass
        dout = loss_fn.backward()
        model.backward(dout)

        # Update parameters
        model.update_params()
```

**PyTorch Implementation:**
```python
for epoch in range(epochs):
    for data, target in train_loader:
        # Forward pass
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()
```

**Analysis:**
- NumPy: Explicit forward/backward calls, manual gradient flow
- PyTorch: Automatic differentiation, cleaner code
- PyTorch handles gradient accumulation automatically

---

## Implementation Details

### Layer Implementation

**NumPy Dense Layer:**
```python
class Dense:
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, dout):
        self.dweights = np.dot(self.input.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        dinput = np.dot(dout, self.weights.T)
        return dinput
```

**PyTorch Linear Layer:**
```python
# Built-in, but conceptually:
# Forward: F.linear(input, weight, bias)
# Backward: Automatic via autograd
```

**Key Differences:**
1. **NumPy**: We manually implement forward and backward passes
2. **PyTorch**: Autograd automatically computes gradients
3. **NumPy**: Explicit gradient storage and computation
4. **PyTorch**: Gradients attached to tensors

---

### Activation Functions

**NumPy ReLU:**
```python
class ReLU:
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, dout):
        dinput = dout.copy()
        dinput[self.input <= 0] = 0
        return dinput
```

**PyTorch ReLU:**
```python
# Built-in functional API
x = F.relu(x)  # Backward handled automatically
```

**Key Differences:**
- NumPy: Manually implement gradient (1 where x > 0, else 0)
- PyTorch: Automatic gradient computation

---

### Optimizers

**NumPy Adam:**
```python
class Adam:
    def update(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            # Update first moment (momentum)
            self.weight_momentums[i] = (
                self.beta1 * self.weight_momentums[i] +
                (1 - self.beta1) * layer.dweights
            )

            # Update second moment (RMSprop)
            self.weight_cache[i] = (
                self.beta2 * self.weight_cache[i] +
                (1 - self.beta2) * np.square(layer.dweights)
            )

            # Bias correction
            m_corrected = self.weight_momentums[i] / (1 - self.beta1 ** self.t)
            v_corrected = self.weight_cache[i] / (1 - self.beta2 ** self.t)

            # Update weights
            layer.weights -= (
                self.learning_rate * m_corrected /
                (np.sqrt(v_corrected) + self.epsilon)
            )
```

**PyTorch Adam:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Usage:
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Key Differences:**
- NumPy: We implement the full Adam algorithm
- PyTorch: Highly optimized built-in implementation
- NumPy: Educational value in understanding the algorithm
- PyTorch: Production-ready with additional features

---

## Performance Comparison

### Expected Results (10 epochs on MNIST)

| Metric | NumPy | PyTorch (CPU) | PyTorch (GPU) |
|--------|-------|---------------|---------------|
| Test Accuracy | 97-98% | 97-98% | 97-98% |
| Training Time | 180-300s | 60-120s | 10-30s |
| Speedup | 1x | 2-3x | 10-20x |
| Memory Usage | Moderate | Moderate | Higher (GPU) |

### Why is PyTorch Faster?

1. **Optimized C++ Backend**: Core operations in C++/CUDA
2. **Parallel Operations**: Better CPU vectorization
3. **GPU Support**: Massive parallelization on CUDA
4. **Memory Management**: More efficient tensor operations
5. **JIT Compilation**: Runtime optimizations

### Why Use NumPy Implementation?

1. **Learning**: Understand what's happening under the hood
2. **Transparency**: See every matrix multiplication
3. **No Black Box**: Full control over every operation
4. **Debugging**: Easier to trace issues
5. **Educational**: Perfect for teaching/learning ML

---

## Feature Comparison

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| **Automatic Differentiation** | Manual | ✓ Autograd |
| **GPU Support** | ✗ | ✓ CUDA |
| **Dynamic Graphs** | N/A | ✓ Native |
| **Pre-trained Models** | ✗ | ✓ TorchVision |
| **Distributed Training** | ✗ | ✓ DDP |
| **Mixed Precision** | ✗ | ✓ AMP |
| **Model Export** | Custom | ✓ ONNX |
| **Mobile Deployment** | ✗ | ✓ PyTorch Mobile |
| **Debugging** | Easy | Tools available |
| **Learning Curve** | See everything | Learn framework |

---

## Code Complexity

### NumPy Implementation

**Total files**: 6 core modules + utilities
**Lines of code**: ~1,200 lines

Files:
- `layers.py` (100 lines) - Dense layer
- `activations.py` (180 lines) - 4 activation functions
- `losses.py` (120 lines) - Loss functions
- `model.py` (210 lines) - Neural network class
- `optimizers.py` (180 lines) - 3 optimizers
- `utils.py` (220 lines) - Utilities

**Pros:**
- Complete understanding of every operation
- Educational value
- No hidden abstractions

**Cons:**
- More code to maintain
- Easier to introduce bugs
- Slower execution

### PyTorch Implementation

**Total files**: 2 files
**Lines of code**: ~250 lines

Files:
- `pytorch_model.py` (150 lines) - Model + training
- `train_pytorch.py` (100 lines) - Training script

**Pros:**
- Concise and readable
- Well-tested framework
- Production-ready
- GPU acceleration

**Cons:**
- Less transparency
- Need to learn framework API
- Some operations are "magic"

---

## When to Use Each

### Use NumPy Implementation When:

1. **Learning**: Understanding neural networks from scratch
2. **Teaching**: Explaining how neural networks work
3. **Prototyping**: Quick algorithm experiments
4. **Research**: Testing novel ideas at algorithmic level
5. **Debugging**: Need to understand exact computations

### Use PyTorch Implementation When:

1. **Production**: Deploying real applications
2. **Performance**: Need speed and GPU acceleration
3. **Scale**: Large models or datasets
4. **Research**: Using pre-trained models, transfer learning
5. **Collaboration**: Working with standard ML frameworks
6. **Advanced Features**: Need distributed training, mixed precision, etc.

---

## Running the Comparison

### Install Dependencies

```bash
# For NumPy implementation
pip install numpy scikit-learn matplotlib

# For PyTorch implementation
pip install torch torchvision

# For comparison
pip install numpy torch scikit-learn matplotlib
```

### Run Individual Implementations

```bash
# NumPy version
python train_numpy.py

# PyTorch version
python train_pytorch.py
```

### Run Direct Comparison

```bash
python compare_implementations.py
```

This will:
- Train both models on same data
- Generate comparison plots
- Report accuracy and speed differences
- Create side-by-side visualizations

---

## Key Takeaways

1. **Both implementations achieve similar accuracy** (~97-98% on MNIST)

2. **PyTorch is faster** but NumPy is more educational

3. **NumPy version shows the math** explicitly - great for learning

4. **PyTorch version shows best practices** - great for production

5. **Understanding both** makes you a better ML engineer:
   - Know what's happening under the hood (NumPy)
   - Know how to build efficiently (PyTorch)

6. **The algorithms are the same**, just different implementations

7. **Your choice depends on your goal**:
   - Learning → NumPy
   - Building → PyTorch

---

## Next Steps

### After Understanding Both:

1. **Compare other architectures**: CNNs, RNNs
2. **Implement advanced features**: Dropout, BatchNorm
3. **Try other frameworks**: TensorFlow, JAX
4. **Build real projects**: Use PyTorch for production
5. **Teach others**: Use NumPy for explanation

---

## Conclusion

The NumPy implementation gives you deep understanding, while PyTorch gives you powerful tools. Both are valuable:

- **Learn with NumPy** to understand the fundamentals
- **Build with PyTorch** to create real applications
- **Compare both** to appreciate the engineering
- **Master both** to become an ML expert

The best ML engineers understand both the theory (NumPy) and the practice (PyTorch).
