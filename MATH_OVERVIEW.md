# Neural Network Mathematics: A Practical Guide

This document explains the mathematics behind feedforward neural networks and backpropagation. Focus is on intuition and matrix operations rather than rigorous proofs.

---

## 1. What is a Feedforward Neural Network?

A **feedforward neural network** (also called a multilayer perceptron) is a function approximator that learns to map inputs to outputs through multiple layers of transformations.

**Architecture:**
```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer → Prediction
  X   →      h₁       →       h₂      → ... →      ŷ       →    Class
```

Each layer performs:
1. **Linear transformation**: Multiply by weights, add biases
2. **Nonlinear activation**: Apply an activation function

---

## 2. Forward Propagation

Forward propagation computes the network's output given an input.

### Dense (Fully Connected) Layer

For a layer with input `X` (shape: `[batch_size, n_in]`):

```
Z = X · W + b
```

Where:
- `W`: Weight matrix (shape: `[n_in, n_out]`)
- `b`: Bias vector (shape: `[n_out]`)
- `Z`: Pre-activation output (shape: `[batch_size, n_out]`)

### Activation Functions

After computing `Z`, we apply a nonlinear activation function:

**ReLU (Rectified Linear Unit):**
```
f(z) = max(0, z)
```
- Most common in hidden layers
- Simple, effective, helps with gradient flow

**Sigmoid:**
```
f(z) = 1 / (1 + e^(-z))
```
- Outputs between 0 and 1
- Historically popular, now less common

**Softmax (for output layer):**
```
f(z_i) = e^(z_i) / Σⱼ e^(z_j)
```
- Converts logits to probabilities
- Sum of outputs equals 1
- Used for multi-class classification

### Complete Forward Pass Example

For a 2-layer network:

```
# Layer 1
Z₁ = X · W₁ + b₁
A₁ = ReLU(Z₁)

# Layer 2 (output)
Z₂ = A₁ · W₂ + b₂
A₂ = Softmax(Z₂)  # This is ŷ (predicted probabilities)
```

**Shape tracking example** (MNIST):
- Input: X → [batch_size, 784]
- Layer 1: W₁ → [784, 128], Z₁ → [batch_size, 128], A₁ → [batch_size, 128]
- Layer 2: W₂ → [128, 10], Z₂ → [batch_size, 10], A₂ → [batch_size, 10]

---

## 3. Loss Function

We need to measure how wrong our predictions are.

### Cross-Entropy Loss (for classification)

```
L = -1/N Σᵢ Σⱼ yᵢⱼ · log(ŷᵢⱼ)
```

Where:
- `N`: Batch size
- `yᵢⱼ`: True label (one-hot encoded)
- `ŷᵢⱼ`: Predicted probability

**Intuition:**
- If we predict high probability for the correct class, loss is low
- If we predict low probability for the correct class, loss is high

**Practical form:** For single-label classification:
```
L = -1/N Σᵢ log(ŷᵢ,true_class)
```

---

## 4. Backpropagation (The Heart of Learning)

**Goal:** Compute how much each weight and bias contributed to the error, so we can adjust them.

**Key Idea:** Use the chain rule to propagate gradients backward through the network.

### Chain Rule Reminder

If `y = f(g(x))`, then:
```
dy/dx = (dy/dg) · (dg/dx)
```

### Backpropagation Algorithm

We compute gradients layer by layer, moving backward from output to input.

#### Step 1: Gradient of Loss w.r.t. Output

For cross-entropy + softmax, the gradient simplifies to:
```
dL/dZ₂ = ŷ - y
```
Where:
- `ŷ`: Predicted probabilities (shape: `[batch_size, n_classes]`)
- `y`: True labels (one-hot, shape: `[batch_size, n_classes]`)

**This is remarkably simple!** The gradient is just the prediction error.

#### Step 2: Gradient w.r.t. Weights and Biases (Output Layer)

From `Z₂ = A₁ · W₂ + b₂`, we use chain rule:

```
dL/dW₂ = A₁ᵀ · (dL/dZ₂)
dL/db₂ = sum(dL/dZ₂, axis=0)
```

**Shape check:**
- `A₁ᵀ`: [128, batch_size]
- `dL/dZ₂`: [batch_size, 10]
- `dL/dW₂`: [128, 10] ✓ (same shape as W₂)

#### Step 3: Propagate Gradient to Previous Layer

To continue backward, we need gradient w.r.t. `A₁`:

```
dL/dA₁ = (dL/dZ₂) · W₂ᵀ
```

#### Step 4: Gradient Through Activation (ReLU)

ReLU derivative:
```
f'(z) = 1 if z > 0, else 0
```

So:
```
dL/dZ₁ = dL/dA₁ ⊙ (Z₁ > 0)
```
Where `⊙` is element-wise multiplication.

#### Step 5: Gradient w.r.t. Weights and Biases (Hidden Layer)

```
dL/dW₁ = Xᵀ · (dL/dZ₁)
dL/db₁ = sum(dL/dZ₁, axis=0)
```

### Summary of Backpropagation Flow

```
Loss
  ↓
dL/dZ₂ = ŷ - y
  ↓
dL/dW₂ = A₁ᵀ · dL/dZ₂
dL/db₂ = sum(dL/dZ₂, axis=0)
dL/dA₁ = dL/dZ₂ · W₂ᵀ
  ↓
dL/dZ₁ = dL/dA₁ ⊙ ReLU'(Z₁)
  ↓
dL/dW₁ = Xᵀ · dL/dZ₁
dL/db₁ = sum(dL/dZ₁, axis=0)
```

---

## 5. Optimization (Gradient Descent)

Once we have gradients, we update parameters to reduce loss.

### Stochastic Gradient Descent (SGD)

```
W = W - η · dL/dW
b = b - η · dL/db
```

Where `η` (eta) is the **learning rate** (e.g., 0.01).

### SGD with Momentum

Momentum adds "velocity" to updates, smoothing out oscillations:

```
v_W = β · v_W + dL/dW
W = W - η · v_W
```

Where `β` is the momentum coefficient (typically 0.9).

### Adam (Adaptive Moment Estimation)

Adam combines momentum with adaptive learning rates per parameter. More complex but often faster convergence.

---

## 6. Training Loop

Putting it all together:

```
For each epoch:
    For each mini-batch (X_batch, y_batch):
        1. Forward pass: compute predictions ŷ
        2. Compute loss: L = CrossEntropy(ŷ, y)
        3. Backward pass: compute all gradients
        4. Update parameters: W = W - η · dL/dW

    Compute validation loss and accuracy
```

---

## Key Takeaways

1. **Forward pass**: Multiply by weights, add bias, apply activation
2. **Backward pass**: Use chain rule to compute gradients layer-by-layer
3. **Update**: Move parameters in direction that reduces loss
4. **Repeat**: Over many mini-batches and epochs, the network learns

The beauty is that this same algorithm works for networks of any depth with any differentiable operations.

---

## Matrix Shape Debugging Tips

When implementing, always check shapes:
- Forward: `[batch, n_in] · [n_in, n_out] = [batch, n_out]`
- Backward (dW): `[n_in, batch]ᵀ · [batch, n_out] = [n_in, n_out]`
- Backward (propagate): `[batch, n_out] · [n_out, n_in]ᵀ = [batch, n_in]`

If shapes don't match, check for missing transposes.
