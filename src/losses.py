"""
Loss function implementations.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.

    When used with softmax activation, provides the combined gradient:
    dL/dZ = predictions - targets (very clean!)
    """

    def __init__(self):
        self.predictions = None
        self.targets = None
        self.loss = None

    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss.

        Parameters
        ----------
        predictions : ndarray of shape (batch_size, n_classes)
            Predicted probabilities (should sum to 1 per sample)
        targets : ndarray of shape (batch_size, n_classes) or (batch_size,)
            True labels (one-hot encoded or class indices)

        Returns
        -------
        loss : float
            Average cross-entropy loss
        """
        batch_size = predictions.shape[0]

        self.predictions = predictions
        self.targets = targets

        if targets.ndim == 1:
            targets_one_hot = np.zeros_like(predictions)
            targets_one_hot[np.arange(batch_size), targets] = 1
            self.targets = targets_one_hot
        else:
            targets_one_hot = targets

        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        correct_confidences = np.sum(predictions_clipped * targets_one_hot, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        self.loss = np.mean(negative_log_likelihoods)

        return self.loss

    def backward(self):
        """
        Compute gradient of loss w.r.t. predictions.

        For softmax + cross-entropy, this simplifies to:
        gradient = predictions - targets

        Returns
        -------
        dinput : ndarray of shape (batch_size, n_classes)
            Gradient of loss w.r.t. predictions
        """
        batch_size = self.predictions.shape[0]

        dinput = self.predictions - self.targets

        dinput = dinput / batch_size

        return dinput


class MSELoss:
    """
    Mean Squared Error loss.

    Useful for regression tasks or comparison purposes.
    """

    def __init__(self):
        self.predictions = None
        self.targets = None
        self.loss = None

    def forward(self, predictions, targets):
        """
        Compute MSE loss.

        Parameters
        ----------
        predictions : ndarray
            Predicted values
        targets : ndarray
            True values

        Returns
        -------
        loss : float
            Mean squared error
        """
        self.predictions = predictions
        self.targets = targets

        self.loss = np.mean(np.square(predictions - targets))

        return self.loss

    def backward(self):
        """
        Compute gradient of loss w.r.t. predictions.

        Returns
        -------
        dinput : ndarray
            Gradient of loss w.r.t. predictions
        """
        batch_size = self.predictions.shape[0]

        dinput = 2 * (self.predictions - self.targets) / batch_size

        return dinput
