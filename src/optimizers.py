"""
Optimizer implementations.
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        """
        Update parameters for all trainable layers.

        Parameters
        ----------
        layers : list
            List of trainable layers
        """
        for layer in layers:
            layer.weights -= self.learning_rate * layer.dweights
            layer.bias -= self.learning_rate * layer.dbias


class SGDMomentum:
    """
    SGD with momentum.

    Momentum helps accelerate convergence and smooth out updates.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate
    momentum : float, default=0.9
        Momentum coefficient (typically 0.9 or 0.99)
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_momentums = []
        self.bias_momentums = []

    def update(self, layers):
        """
        Update parameters for all trainable layers using momentum.

        Parameters
        ----------
        layers : list
            List of trainable layers
        """
        if not self.weight_momentums:
            self.weight_momentums = [np.zeros_like(layer.weights) for layer in layers]
            self.bias_momentums = [np.zeros_like(layer.bias) for layer in layers]

        for i, layer in enumerate(layers):
            self.weight_momentums[i] = (
                self.momentum * self.weight_momentums[i] +
                (1 - self.momentum) * layer.dweights
            )

            self.bias_momentums[i] = (
                self.momentum * self.bias_momentums[i] +
                (1 - self.momentum) * layer.dbias
            )

            layer.weights -= self.learning_rate * self.weight_momentums[i]
            layer.bias -= self.learning_rate * self.bias_momentums[i]


class Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).

    Combines momentum with adaptive learning rates per parameter.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate
    beta1 : float, default=0.9
        Exponential decay rate for first moment estimates
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.weight_momentums = []
        self.bias_momentums = []
        self.weight_cache = []
        self.bias_cache = []
        self.t = 0

    def update(self, layers):
        """
        Update parameters for all trainable layers using Adam.

        Parameters
        ----------
        layers : list
            List of trainable layers
        """
        if not self.weight_momentums:
            self.weight_momentums = [np.zeros_like(layer.weights) for layer in layers]
            self.bias_momentums = [np.zeros_like(layer.bias) for layer in layers]
            self.weight_cache = [np.zeros_like(layer.weights) for layer in layers]
            self.bias_cache = [np.zeros_like(layer.bias) for layer in layers]

        self.t += 1

        for i, layer in enumerate(layers):
            self.weight_momentums[i] = (
                self.beta1 * self.weight_momentums[i] +
                (1 - self.beta1) * layer.dweights
            )
            self.bias_momentums[i] = (
                self.beta1 * self.bias_momentums[i] +
                (1 - self.beta1) * layer.dbias
            )

            self.weight_cache[i] = (
                self.beta2 * self.weight_cache[i] +
                (1 - self.beta2) * np.square(layer.dweights)
            )
            self.bias_cache[i] = (
                self.beta2 * self.bias_cache[i] +
                (1 - self.beta2) * np.square(layer.dbias)
            )

            weight_momentums_corrected = self.weight_momentums[i] / (1 - self.beta1 ** self.t)
            bias_momentums_corrected = self.bias_momentums[i] / (1 - self.beta1 ** self.t)

            weight_cache_corrected = self.weight_cache[i] / (1 - self.beta2 ** self.t)
            bias_cache_corrected = self.bias_cache[i] / (1 - self.beta2 ** self.t)

            layer.weights -= (
                self.learning_rate * weight_momentums_corrected /
                (np.sqrt(weight_cache_corrected) + self.epsilon)
            )
            layer.bias -= (
                self.learning_rate * bias_momentums_corrected /
                (np.sqrt(bias_cache_corrected) + self.epsilon)
            )
