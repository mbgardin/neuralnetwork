"""
Neural Network model implementation.
"""

import numpy as np


class NeuralNetwork:
    """
    Feedforward neural network with flexible architecture.

    Example
    -------
    >>> from src.layers import Dense
    >>> from src.activations import ReLU, Softmax
    >>> from src.losses import CrossEntropyLoss
    >>> from src.optimizers import SGD
    >>>
    >>> model = NeuralNetwork()
    >>> model.add(Dense(784, 128))
    >>> model.add(ReLU())
    >>> model.add(Dense(128, 10))
    >>> model.add(Softmax())
    >>> model.compile(loss=CrossEntropyLoss(), optimizer=SGD(learning_rate=0.01))
    """

    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.trainable_layers = []

    def add(self, layer):
        """
        Add a layer to the network.

        Parameters
        ----------
        layer : Layer or Activation
            Layer to add to the network
        """
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """
        Compile the model with loss function and optimizer.

        Parameters
        ----------
        loss : Loss
            Loss function instance
        optimizer : Optimizer
            Optimizer instance
        """
        self.loss_function = loss
        self.optimizer = optimizer

        self.trainable_layers = [layer for layer in self.layers if hasattr(layer, 'weights')]

    def forward(self, X, training=True):
        """
        Forward pass through all layers.

        Parameters
        ----------
        X : ndarray of shape (batch_size, n_features)
            Input data
        training : bool, default=True
            Whether in training mode (for dropout, etc.)

        Returns
        -------
        output : ndarray
            Network output
        """
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, dout):
        """
        Backward pass through all layers.

        Parameters
        ----------
        dout : ndarray
            Gradient of loss w.r.t. network output
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_params(self):
        """
        Update parameters of all trainable layers using the optimizer.
        """
        self.optimizer.update(self.trainable_layers)

    def predict(self, X):
        """
        Make predictions (forward pass in inference mode).

        Parameters
        ----------
        X : ndarray
            Input data

        Returns
        -------
        predictions : ndarray
            Network predictions
        """
        return self.forward(X, training=False)

    def train_step(self, X_batch, y_batch):
        """
        Perform one training step (forward, loss, backward, update).

        Parameters
        ----------
        X_batch : ndarray
            Input batch
        y_batch : ndarray
            Target batch

        Returns
        -------
        loss : float
            Loss value for this batch
        """
        predictions = self.forward(X_batch, training=True)

        loss = self.loss_function.forward(predictions, y_batch)

        dout = self.loss_function.backward()

        self.backward(dout)

        self.update_params()

        return loss

    def evaluate(self, X, y):
        """
        Evaluate model on dataset.

        Parameters
        ----------
        X : ndarray
            Input data
        y : ndarray
            True labels

        Returns
        -------
        loss : float
            Average loss
        accuracy : float
            Classification accuracy
        """
        predictions = self.predict(X)

        loss = self.loss_function.forward(predictions, y)

        if y.ndim == 1:
            y_pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(y_pred_classes == y)
        else:
            y_pred_classes = np.argmax(predictions, axis=1)
            y_true_classes = np.argmax(y, axis=1)
            accuracy = np.mean(y_pred_classes == y_true_classes)

        return loss, accuracy

    def get_config(self):
        """
        Get model configuration summary.

        Returns
        -------
        config : dict
            Model configuration
        """
        config = {
            'num_layers': len(self.layers),
            'trainable_layers': len(self.trainable_layers),
            'total_params': sum(
                layer.weights.size + layer.bias.size
                for layer in self.trainable_layers
            )
        }
        return config

    def summary(self):
        """
        Print model summary.
        """
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)

        total_params = 0

        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__

            if hasattr(layer, 'weights'):
                n_params = layer.weights.size + layer.bias.size
                total_params += n_params
                shape_info = f"{layer.n_inputs} → {layer.n_outputs}"
                print(f"Layer {i+1:2d}: {layer_name:15s} {shape_info:20s} | Params: {n_params:,}")
            else:
                print(f"Layer {i+1:2d}: {layer_name:15s} {'(activation)':20s} | Params: 0")

        print("=" * 60)
        print(f"Total trainable parameters: {total_params:,}")
        print("=" * 60)
