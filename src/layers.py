"""
Layer implementations for neural network.
"""

import numpy as np


class Dense:
    """
    Fully connected (dense) layer.

    Performs the operation: output = input @ weights + bias

    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output features
    weight_init : str, default='he'
        Weight initialization method ('he', 'xavier', 'random')
    """

    def __init__(self, n_inputs, n_outputs, weight_init='he'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        if weight_init == 'he':
            self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        elif weight_init == 'xavier':
            self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(1.0 / n_inputs)
        else:
            self.weights = np.random.randn(n_inputs, n_outputs) * 0.01

        self.bias = np.zeros((1, n_outputs))

        self.input = None
        self.output = None

        self.dweights = None
        self.dbias = None

    def forward(self, input_data):
        """
        Forward pass through the layer.

        Parameters
        ----------
        input_data : ndarray of shape (batch_size, n_inputs)
            Input data

        Returns
        -------
        output : ndarray of shape (batch_size, n_outputs)
            Output of the layer
        """
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, dout):
        """
        Backward pass through the layer.

        Parameters
        ----------
        dout : ndarray of shape (batch_size, n_outputs)
            Gradient of loss with respect to layer output

        Returns
        -------
        dinput : ndarray of shape (batch_size, n_inputs)
            Gradient of loss with respect to layer input
        """
        self.dweights = np.dot(self.input.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)

        dinput = np.dot(dout, self.weights.T)

        return dinput

    def get_params(self):
        """Return layer parameters."""
        return {'weights': self.weights, 'bias': self.bias}

    def get_gradients(self):
        """Return parameter gradients."""
        return {'weights': self.dweights, 'bias': self.dbias}

    def set_params(self, params):
        """Set layer parameters."""
        self.weights = params['weights']
        self.bias = params['bias']
