"""
PyTorch implementation of the same neural network architecture.

This allows direct comparison with the NumPy implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


class MNISTNet(nn.Module):
    """
    PyTorch neural network matching the NumPy implementation.

    Architecture: 784 → 128 → 64 → 10
    """

    def __init__(self):
        super(MNISTNet, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization (matching NumPy implementation)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 784)
            Input data

        Returns
        -------
        output : torch.Tensor of shape (batch_size, 10)
            Output logits (before softmax)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def summary(self):
        """Print model summary."""
        print("=" * 60)
        print("PyTorch Model Summary")
        print("=" * 60)

        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:20s} | Shape: {str(list(param.shape)):20s} | Params: {num_params:,}")

        print("=" * 60)
        print(f"Total trainable parameters: {total_params:,}")
        print("=" * 60)


def train_pytorch_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=15,
    verbose=True
):
    """
    Train the PyTorch model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    optimizer : torch.optim.Optimizer
        Optimizer
    device : str
        Device to train on ('cpu' or 'cuda')
    epochs : int
        Number of epochs
    verbose : bool
        Whether to print progress

    Returns
    -------
    history : dict
        Training history
    """
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if verbose:
        print("\nStarting PyTorch training...")
        print(f"Device: {device}")
        print(f"Epochs: {epochs}")
        print("=" * 70)

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    if verbose:
        print("=" * 70)
        print("Training complete!")

    return history


def evaluate_pytorch_model(model, test_loader, device):
    """
    Evaluate the PyTorch model on test set.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    test_loader : DataLoader
        Test data loader
    device : str
        Device

    Returns
    -------
    test_loss : float
        Test loss
    test_acc : float
        Test accuracy
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / total

    return test_loss, test_acc


def prepare_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    """
    Prepare PyTorch DataLoaders from NumPy arrays.

    Parameters
    ----------
    X_train, y_train : ndarray
        Training data
    X_val, y_val : ndarray
        Validation data
    X_test, y_test : ndarray
        Test data
    batch_size : int
        Batch size

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    model = MNISTNet()
    model.summary()

    print("\nPyTorch model architecture matches NumPy implementation:")
    print("  Layer 1: 784 → 128 (ReLU)")
    print("  Layer 2: 128 → 64 (ReLU)")
    print("  Layer 3: 64 → 10 (output)")
    print("\nTo train, use train_pytorch.py")
