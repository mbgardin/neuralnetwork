"""
Utility functions for data loading, preprocessing, and metrics.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_mnist(data_dir='./data', normalize=True, flatten=True):
    """
    Load MNIST dataset.

    Parameters
    ----------
    data_dir : str
        Directory to cache the dataset
    normalize : bool, default=True
        Whether to normalize pixel values to [0, 1]
    flatten : bool, default=True
        Whether to flatten images to 1D vectors

    Returns
    -------
    X_train : ndarray
        Training images
    X_test : ndarray
        Test images
    y_train : ndarray
        Training labels
    y_test : ndarray
        Test labels
    """
    os.makedirs(data_dir, exist_ok=True)

    print("Loading MNIST dataset...")

    mnist = fetch_openml('mnist_784', version=1, cache=True, data_home=data_dir, parser='auto')

    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')

    if isinstance(X, np.ndarray):
        X = X
    else:
        X = X.to_numpy()

    if isinstance(y, np.ndarray):
        y = y
    else:
        y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    print(f"Loaded MNIST: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    return X_train, X_test, y_train, y_test


def create_mini_batches(X, y, batch_size=32, shuffle=True):
    """
    Create mini-batches from dataset.

    Parameters
    ----------
    X : ndarray
        Input data
    y : ndarray
        Target labels
    batch_size : int, default=32
        Size of each mini-batch
    shuffle : bool, default=True
        Whether to shuffle data before creating batches

    Yields
    ------
    X_batch : ndarray
        Input batch
    y_batch : ndarray
        Target batch
    """
    n_samples = X.shape[0]

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        yield X_batch, y_batch


def one_hot_encode(y, num_classes=10):
    """
    Convert class labels to one-hot encoded vectors.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels
    num_classes : int, default=10
        Number of classes

    Returns
    -------
    y_one_hot : ndarray of shape (n_samples, num_classes)
        One-hot encoded labels
    """
    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, num_classes))
    y_one_hot[np.arange(n_samples), y] = 1
    return y_one_hot


def compute_accuracy(predictions, targets):
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions : ndarray
        Predicted class probabilities or logits
    targets : ndarray
        True labels (class indices or one-hot)

    Returns
    -------
    accuracy : float
        Classification accuracy
    """
    pred_classes = np.argmax(predictions, axis=1)

    if targets.ndim == 1:
        true_classes = targets
    else:
        true_classes = np.argmax(targets, axis=1)

    accuracy = np.mean(pred_classes == true_classes)

    return accuracy


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Parameters
    ----------
    patience : int, default=5
        Number of epochs to wait for improvement
    min_delta : float, default=0.001
        Minimum change to qualify as improvement
    """

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation loss

        Returns
        -------
        should_stop : bool
            Whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


class History:
    """
    Track training history.
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def update(self, train_loss, train_acc, val_loss=None, val_acc=None):
        """
        Update history with new metrics.

        Parameters
        ----------
        train_loss : float
            Training loss
        train_acc : float
            Training accuracy
        val_loss : float, optional
            Validation loss
        val_acc : float, optional
            Validation accuracy
        """
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)

    def get_history(self):
        """
        Get complete history dictionary.

        Returns
        -------
        history : dict
            Training history
        """
        return self.history
