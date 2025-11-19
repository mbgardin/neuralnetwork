"""
Activation function implementations.
"""

import numpy as np


class ReLU:
    """
    Rectified Linear Unit activation function.

    f(x) = max(0, x)
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Forward pass.

        Parameters
        ----------
        input_data : ndarray
            Input data

        Returns
        -------
        output : ndarray
            ReLU(input_data)
        """
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, dout):
        """
        Backward pass.

        Gradient is 1 where input > 0, else 0.

        Parameters
        ----------
        dout : ndarray
            Gradient of loss w.r.t. output

        Returns
        -------
        dinput : ndarray
            Gradient of loss w.r.t. input
        """
        dinput = dout.copy()
        dinput[self.input <= 0] = 0
        return dinput


class Sigmoid:
    """
    Sigmoid activation function.

    f(x) = 1 / (1 + exp(-x))
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Forward pass.

        Parameters
        ----------
        input_data : ndarray
            Input data

        Returns
        -------
        output : ndarray
            Sigmoid(input_data)
        """
        self.input = input_data
        self.output = 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
        return self.output

    def backward(self, dout):
        """
        Backward pass.

        Gradient is sigmoid(x) * (1 - sigmoid(x)).

        Parameters
        ----------
        dout : ndarray
            Gradient of loss w.r.t. output

        Returns
        -------
        dinput : ndarray
            Gradient of loss w.r.t. input
        """
        sigmoid_grad = self.output * (1 - self.output)
        dinput = dout * sigmoid_grad
        return dinput


class Tanh:
    """
    Hyperbolic tangent activation function.

    f(x) = tanh(x)
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Forward pass.

        Parameters
        ----------
        input_data : ndarray
            Input data

        Returns
        -------
        output : ndarray
            Tanh(input_data)
        """
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, dout):
        """
        Backward pass.

        Gradient is 1 - tanh(x)^2.

        Parameters
        ----------
        dout : ndarray
            Gradient of loss w.r.t. output

        Returns
        -------
        dinput : ndarray
            Gradient of loss w.r.t. input
        """
        tanh_grad = 1 - np.power(self.output, 2)
        dinput = dout * tanh_grad
        return dinput


class Softmax:
    """
    Softmax activation function.

    Converts logits to probability distribution.
    f(x_i) = exp(x_i) / sum(exp(x_j))
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """
        Forward pass.

        Uses numerically stable softmax with max subtraction.

        Parameters
        ----------
        input_data : ndarray of shape (batch_size, n_classes)
            Input logits

        Returns
        -------
        output : ndarray of shape (batch_size, n_classes)
            Probability distribution (rows sum to 1)
        """
        self.input = input_data

        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return self.output

    def backward(self, dout):
        """
        Backward pass.

        Note: When used with cross-entropy loss, this backward pass
        is typically not called directly. Instead, we use the combined
        gradient (predictions - targets) for numerical stability.

        Parameters
        ----------
        dout : ndarray
            Gradient of loss w.r.t. output

        Returns
        -------
        dinput : ndarray
            Gradient of loss w.r.t. input
        """
        dinput = np.empty_like(dout)

        for i, (single_output, single_dout) in enumerate(zip(self.output, dout)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            dinput[i] = np.dot(jacobian, single_dout)

        return dinput
