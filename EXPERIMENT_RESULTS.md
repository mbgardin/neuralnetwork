# Experiment Results

This document will record results from your experiments.

## Template

Copy this template for each experiment:

```markdown
### Experiment: [Name]
**Date**: YYYY-MM-DD
**Script**: [e.g., train_numpy.py]

**Configuration**:
- Architecture: [e.g., 784→128→64→10]
- Optimizer: [e.g., Adam(lr=0.001)]
- Batch size: [e.g., 128]
- Epochs: [e.g., 15]

**Results**:
- Training accuracy: XX.XX%
- Validation accuracy: XX.XX%
- Test accuracy: XX.XX%
- Training time: XXX seconds
- Final loss: X.XXXX

**Observations**:
- [What did you notice?]
- [Any interesting patterns?]
- [What would you try next?]

**Plots**: [Path to saved plots]
```

---

## Example Results

### Experiment: Baseline Small Network
**Date**: 2024-11-19
**Script**: toy_experiment.py

**Configuration**:
- Architecture: 2→16→3 (spiral classification)
- Optimizer: SGD(lr=0.1)
- Epochs: 100

**Expected Results**:
- Test accuracy: ~85-90%
- Fast training (~10 seconds)
- Smooth decision boundaries

**Observations**:
- SGD converges but slower than Adam
- Simple architecture sufficient for toy problem
- Good for quick verification

---

### Experiment: Baseline Adam
**Date**: 2024-11-19
**Script**: toy_experiment.py

**Configuration**:
- Architecture: 2→32→16→3 (spiral classification)
- Optimizer: Adam(lr=0.01)
- Epochs: 100

**Expected Results**:
- Test accuracy: ~92-95%
- Fast convergence
- Complex decision boundaries

**Observations**:
- Adam converges faster than SGD
- Deeper network captures spiral pattern better
- More parameters = more expressive

---

### Experiment: MNIST Baseline
**Date**: TBD
**Script**: train_numpy.py

**Configuration**:
- Architecture: 784→128→64→10
- Optimizer: Adam(lr=0.001)
- Batch size: 128
- Epochs: 15

**Target Results**:
- Training accuracy: ~98-99%
- Validation accuracy: ~97-98%
- Test accuracy: ~97-98%
- Training time: ~3-5 minutes (CPU)
- Final loss: ~0.10-0.15

**What to look for**:
- Loss should decrease steadily
- Train/val accuracy should be close (no overfitting)
- Convergence around epoch 10-12

---

## Your Results

### Experiment 1: [Your Title]

[Fill in your results here]

---

## Comparison Table

| Experiment | Architecture | Optimizer | Test Acc | Time | Notes |
|------------|--------------|-----------|----------|------|-------|
| Toy-SGD    | 2→16→3      | SGD       | ~88%     | 10s  | Baseline |
| Toy-Adam   | 2→32→16→3   | Adam      | ~94%     | 15s  | Better |
| MNIST-1    | 784→128→10  | Adam      | TBD      | TBD  | |
| MNIST-2    | 784→256→128→10 | Adam   | TBD      | TBD  | Deeper |

---

## Lessons Learned

### What worked well:
- [List successful approaches]
- [What configurations gave best results?]

### What didn't work:
- [List failed attempts]
- [What would you avoid next time?]

### Surprises:
- [Unexpected results]
- [Interesting patterns]

### Next steps:
- [What to try next]
- [Ideas for improvement]

---

## Tips for Recording Results

1. **Be specific**: Record exact hyperparameters
2. **Save plots**: Keep visual records
3. **Note observations**: What did you learn?
4. **Compare fairly**: Use same random seed for comparison
5. **Track time**: Important for practical applications

---

## Quick Commands

```bash
# Run and time an experiment
time python train_numpy.py

# Compare multiple runs
python experiment.py

# Generate visualizations
python visualize_training.py
python toy_experiment.py
```

---

Start experimenting and record your results above!
